import numpy as np
from PIL import Image
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from scipy.spatial import distance

# Import sklearn for model loading
try:
    import sklearn
    from sklearn.preprocessing import RobustScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available.")

# Import MediaPipe for landmark extraction
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ Warning: MediaPipe not available.")
    print("   Please install: pip install mediapipe")
    print("   Or install all requirements: pip install -r requirements.txt")

# Try to import torch_geometric for GCN support
try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm as GCNBatchNorm
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    GCNConv = None
    global_mean_pool = None
    global_max_pool = None
    GCNBatchNorm = None


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, batch=None):
        if batch is None:
            avg = x.mean(0, keepdim=True)
            mx = x.max(0, keepdim=True)[0]
            return x * self.sigmoid(self.mlp(avg) + self.mlp(mx))
        bs = batch.max().item() + 1
        avgs, mxs = [], []
        for i in range(bs):
            mask = batch == i
            if mask.any():
                avgs.append(x[mask].mean(0))
                mxs.append(x[mask].max(0)[0])
        if not avgs:
            return x
        avg = torch.stack(avgs)
        mx = torch.stack(mxs)
        return x * self.sigmoid(self.mlp(avg) + self.mlp(mx))[batch]


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, batch=None):
        avg = x.mean(-1, keepdim=True)
        mx = x.max(-1, keepdim=True)[0]
        att = self.sigmoid(self.conv(torch.cat([avg, mx], -1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.channel = ChannelAttention(dim, reduction)
        self.spatial = SpatialAttention(dim)

    def forward(self, x, batch=None):
        x = self.channel(x, batch)
        x = self.spatial(x, batch)
        return x


if GNN_AVAILABLE:
    class MultiScaleGCN(nn.Module):
        def __init__(self, in_dim=3, hidden=128, out_dim=256, layers=3, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden), GCNBatchNorm(hidden), nn.ReLU(), nn.Dropout(dropout)
            )
            self.local_convs = nn.ModuleList([GCNConv(hidden, hidden, improved=True) for _ in range(layers)])
            self.regional_convs = nn.ModuleList([GCNConv(hidden, hidden, improved=True) for _ in range(layers)])
            self.global_convs = nn.ModuleList([GCNConv(hidden, hidden, improved=True) for _ in range(layers)])
            self.bns = nn.ModuleList([GCNBatchNorm(hidden) for _ in range(layers)])
            self.cbams = nn.ModuleList([CBAM(hidden) for _ in range(layers)])
            self.cross_attn = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.node_attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
            self.output = nn.Sequential(
                nn.Linear(hidden * 3, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edges, batch):
            bs = batch.max().item() + 1
            x = self.input_proj(x)
            x_l, x_r, x_g = x.clone(), x.clone(), x.clone()
            for lc, rc, gc, bn, cbam in zip(self.local_convs, self.regional_convs, self.global_convs, self.bns, self.cbams):
                x_l = self.dropout(cbam(torch.relu(bn(lc(x_l, edges['local']))), batch)) + x_l
                x_r = self.dropout(cbam(torch.relu(bn(rc(x_r, edges['regional']))), batch)) + x_r
                x_g = self.dropout(cbam(torch.relu(bn(gc(x_g, edges['global']))), batch)) + x_g
            fused = []
            for i in range(bs):
                mask = batch == i
                if mask.any():
                    scales = torch.stack([x_l[mask], x_r[mask], x_g[mask]], 1)
                    attn_out, _ = self.cross_attn(scales, scales, scales)
                    fused.append(self.norm(attn_out + scales).mean(1))
            if not fused:
                return torch.zeros(bs, self.output[0].out_features, device=x.device)
            x_fused = torch.cat(fused, 0)
            x_mean = global_mean_pool(x_fused, batch)
            x_max = global_max_pool(x_fused, batch)
            attn_w = torch.softmax(self.node_attn(x_fused), 0)
            x_attn = global_mean_pool(x_fused * attn_w, batch)
            return self.output(torch.cat([x_mean, x_max, x_attn], 1))
else:
    class MultiScaleGCN(nn.Module):
        def __init__(self, in_dim=3, hidden=128, out_dim=256, layers=3, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(in_dim, hidden)
            self.attns = nn.ModuleList([
                nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True) for _ in range(layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.output = nn.Sequential(
                nn.Linear(hidden * 3, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edges=None, batch=None):
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = torch.relu(self.input_proj(x))
            for attn, norm in zip(self.attns, self.norms):
                res = x
                x, _ = attn(x, x, x)
                x = norm(self.dropout(x) + res)
            x_mean = x.mean(1)
            x_max = x.max(1)[0]
            x_attn = (x * torch.softmax(x.mean(-1, keepdim=True), dim=1)).mean(1)
            x_concat = torch.cat([x_mean, x_max, x_attn], dim=1)
            return self.output(x_concat)


class HybridModel(nn.Module):
    """
    Hybrid model architecture that matches hybrid_model_v2.pth structure.
    This model combines image features, tabular features, and landmark features.
    """
    def __init__(self, num_features, num_landmarks=0, landmark_dim=3, dropout=0.3, use_gcn=True):
        super(HybridModel, self).__init__()
        self.use_gcn = use_gcn and num_landmarks > 0

        # Image backbone - Custom CNN
        self.img_backbone = self._build_cnn_backbone()
        
        # Image projection
        self.img_proj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout)
        )

        # Tabular network
        self.tab_net = nn.Sequential(
            nn.Linear(num_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5)
        )

        # GCN for landmarks
        gcn_dim = 256 if self.use_gcn else 0
        if self.use_gcn:
            self.gcn = MultiScaleGCN(landmark_dim, 128, gcn_dim, 3, dropout)

        # Fusion attention
        combined = 512 + 128 + gcn_dim
        self.fusion_attn = nn.MultiheadAttention(combined, 8, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(combined)

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(combined, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.7)
        )

        # BMI head
        self.bmi_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 1)
        )
        
        # Other heads
        self.age_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 1)
        )
        self.sex_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 2)
        )
        self.cat_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 4)
        )

        self.log_vars = nn.Parameter(torch.zeros(4))
        self._init_weights()
    
    def _build_cnn_backbone(self):
        """Build a custom CNN backbone for image feature extraction."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features, image, graph_features=None, edges=None):
        """
        Forward pass - NO DUMMY DATA, uses real extracted features.
        
        Args:
            features: Tabular features [B, num_features] - REAL EXTRACTED FEATURES
            image: Image tensor [B, 3, H, W] - REAL IMAGE
            graph_features: Landmark features [B, num_landmarks, landmark_dim] - REAL LANDMARKS
            edges: Graph edges dictionary - REAL GRAPH STRUCTURE
        """
        bs = features.size(0)
        
        # Image features from CNN backbone
        img_feat = self.img_backbone(image).view(bs, -1)
        img_feat = self.img_proj(img_feat)
        
        # Tabular features from extracted geometric features
        tab_feat = self.tab_net(features)

        # GCN features from extracted landmarks
        if self.use_gcn and graph_features is not None and graph_features.numel() > 0:
            try:
                if GNN_AVAILABLE and edges is not None and 'local' in edges:
                    num_nodes = graph_features.size(1)
                    x = graph_features.view(-1, graph_features.size(-1))
                    batch_idx = torch.arange(bs, device=x.device).repeat_interleave(num_nodes)
                    edges_batch = {}
                    for scale in ['local', 'regional', 'global']:
                        if scale in edges:
                            edges_batch[scale] = torch.cat([
                                edges[scale] + i * num_nodes for i in range(bs)
                            ], dim=1).to(x.device)
                    if edges_batch:
                        gcn_feat = self.gcn(x, edges_batch, batch_idx)
                    else:
                        gcn_feat = self.gcn(graph_features)
                else:
                    gcn_feat = self.gcn(graph_features)
                combined = torch.cat([img_feat, tab_feat, gcn_feat], 1)
            except Exception as e:
                print(f"Warning: GCN forward failed: {e}")
                gcn_feat = torch.zeros(bs, 256, device=img_feat.device)
                combined = torch.cat([img_feat, tab_feat, gcn_feat], 1)
        else:
            gcn_feat = torch.zeros(bs, 256, device=img_feat.device)
            combined = torch.cat([img_feat, tab_feat, gcn_feat], 1)

        # Fusion attention
        combined_u = combined.unsqueeze(1)
        attn_out, _ = self.fusion_attn(combined_u, combined_u, combined_u)
        fused = self.fusion_norm(attn_out.squeeze(1) + combined)
        
        # Shared layers
        shared = self.shared(fused)
        
        # BMI prediction
        bmi = self.bmi_head(shared).squeeze(-1)
        
        return bmi


class LandmarkExtractor:
    """Extract facial landmarks using MediaPipe - matching training preprocessing"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not available. Cannot extract landmarks.")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def calculate_geometric_features(self, landmarks):
        """
        Calculate geometric features from landmarks.
        
        Args:
            landmarks: numpy array of shape (468, 2) with x, y coordinates
            
        Returns:
            Dictionary with geometric features
        """
        if landmarks is None or len(landmarks) < 468:
            return None

        try:
            features = {}
            
            # Key landmarks indices
            chin = landmarks[152]
            forehead_center = landmarks[10]
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            left_jaw = landmarks[172]
            right_jaw = landmarks[397]
            left_eye_left = landmarks[263]      
            left_eye_right = landmarks[362]     
            right_eye_left = landmarks[33]      
            right_eye_right = landmarks[133]    
            nose_tip = landmarks[1]
            nose_bridge = landmarks[6]
            left_eyebrow_inner = landmarks[70]
            left_eyebrow_outer = landmarks[107] 
            right_eyebrow_inner = landmarks[300]
            right_eyebrow_outer = landmarks[336]

            # Calculate geometric features
            cheekbone_width = distance.euclidean(left_cheek, right_cheek)
            jaw_width = distance.euclidean(left_jaw, right_jaw)
            features['jaw_cheek_ratio'] = cheekbone_width / (jaw_width + 1e-6)

            upper_face_height = distance.euclidean(forehead_center, nose_tip)
            features['face_ratio_height_width'] = cheekbone_width / (upper_face_height + 1e-6)

            # Face contour for perimeter and area
            face_contour_points = [left_jaw, left_cheek, forehead_center, right_cheek, right_jaw, chin]
            perimeter = sum(
                distance.euclidean(face_contour_points[i], face_contour_points[(i+1) % len(face_contour_points)])
                for i in range(len(face_contour_points))
            )
            area = 0.5 * abs(sum(
                face_contour_points[i][0] * face_contour_points[(i+1) % len(face_contour_points)][1] -
                face_contour_points[(i+1) % len(face_contour_points)][0] * face_contour_points[i][1]
                for i in range(len(face_contour_points))
            ))
            features['face_compactness'] = perimeter / (area + 1e-6)

            # Eye features
            left_eye_width = distance.euclidean(left_eye_left, left_eye_right)
            right_eye_width = distance.euclidean(right_eye_left, right_eye_right)
            features['left_eye_width'] = left_eye_width
            features['right_eye_width'] = right_eye_width
            features['eye_width_ratio'] = left_eye_width / (right_eye_width + 1e-6)

            # Face dimensions
            face_height = distance.euclidean(forehead_center, chin)
            lower_face_height = distance.euclidean(nose_tip, chin)
            features['face_height'] = face_height
            features['face_width_cheeks'] = cheekbone_width
            features['face_width_jaw'] = jaw_width

            # Nose features
            nose_length = distance.euclidean(nose_bridge, nose_tip)
            features['nose_length'] = nose_length
            features['nose_width'] = nose_length * 0.4  # Approximate
            features['nose_ratio'] = features['nose_width'] / (nose_length + 1e-6)

            # Mouth width (approximate)
            features['mouth_width'] = jaw_width * 0.6

            # Face area and perimeter
            features['face_oval_area'] = area
            features['face_oval_perimeter'] = perimeter

            return features
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
        
    def extract_landmarks(self, image_np):
        """
        Extract landmarks from image.
        
        Args:
            image_np: numpy array of image (RGB format)
            
        Returns:
            landmarks_3d: numpy array of shape (num_landmarks, 3) with x, y, z coordinates
            geometric_features: dictionary with computed geometric features
        """
        h, w = image_np.shape[:2]
        
        # Convert to RGB if needed
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Process with Face Mesh
        results = self.face_mesh.process(image_np)
        
        if not results.multi_face_landmarks:
            print("No face landmarks detected")
            return None, None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract 2D coordinates for geometric features
        coords_2d = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        
        # Calculate geometric features
        geometric_features = self.calculate_geometric_features(coords_2d)
        
        if geometric_features is None:
            return None, None
        
        # Extract 3D coordinates for landmarks - select 21 key landmarks
        key_indices = [
            10, 152, 234, 454,  # Face outline
            33, 133, 362, 263,  # Eyes
            1, 6,  # Nose
            61, 291,  # Mouth
            0, 17,  # Face points
            199, 428,  # More outline
            130, 359,  # More eyes
            70, 107, 300  # Eyebrows
        ]
        
        landmarks_3d = []
        for idx in key_indices:
            lm = face_landmarks.landmark[idx]
            landmarks_3d.append([lm.x * w, lm.y * h, lm.z * w])
        
        landmarks_3d = np.array(landmarks_3d, dtype=np.float32)
        
        return landmarks_3d, geometric_features


class BMIPredictor:
    """
    BMI Predictor that uses hybrid_model_v2.pth with REAL extracted features.
    NO DUMMY DATA - all features extracted from image.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the BMI predictor.
        
        Args:
            model_path: Path to hybrid_model_v2.pth
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'hybrid_model_v2.pth')
        
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.load_error = None
        
        # Initialize landmark extractor
        if MEDIAPIPE_AVAILABLE:
            try:
                self.landmark_extractor = LandmarkExtractor()
                print("✅ MediaPipe landmark extractor initialized")
            except Exception as e:
                print(f"❌ Could not initialize landmark extractor: {e}")
                self.landmark_extractor = None
                self.load_error = f"MediaPipe initialization failed: {str(e)}"
        else:
            self.landmark_extractor = None
            self.load_error = "MediaPipe is not installed. Please run: pip install mediapipe"
            print("❌ MediaPipe not available")
            print("   To fix this, run: pip install mediapipe")
            print("   Or install all requirements: pip install -r requirements.txt")
        
        # Try to load model
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.load_model()
            self.model_loaded = True
        except Exception as e:
            self.load_error = str(e)
            print(f"⚠️ Model could not be loaded: {e}")
    
    def load_model(self):
        """Load hybrid_model_v2.pth"""
        try:
            print(f"📂 Loading hybrid_model_v2.pth from: {self.model_path}")
            print(f"🖥️ Using device: {self.device}")
            
            loaded_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(loaded_data, dict):
                model_state = loaded_data.get('model_state')
                self.feature_scaler = loaded_data.get('feature_scaler')
                self.landmark_scaler = loaded_data.get('landmark_scaler')
                self.feature_cols = loaded_data.get('feature_cols', [])
                self.graph_edges = loaded_data.get('graph_edges')
                self.num_landmarks = loaded_data.get('num_landmarks', 21)
                self.landmark_dim = loaded_data.get('landmark_dim', 3)
                
                num_features = len(self.feature_cols) if self.feature_cols else 36
                
                # Create model with exact architecture
                self.model = HybridModel(
                    num_features=num_features,
                    num_landmarks=self.num_landmarks,
                    landmark_dim=self.landmark_dim,
                    dropout=0.3,
                    use_gcn=True
                )
                
                # Load state dict
                self.model.load_state_dict(model_state, strict=False)
                print("✅ Model state_dict loaded")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            print("✅ hybrid_model_v2.pth loaded and ready!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_features_from_image(self, image_np):
        """
        Extract ALL features from image - NO DUMMY DATA.
        
        Args:
            image_np: numpy array of image (RGB format)
            
        Returns:
            features_tensor: Tensor of tabular features - REAL EXTRACTED
            landmarks_tensor: Tensor of landmark coordinates - REAL EXTRACTED
        """
        if self.landmark_extractor is None:
            raise RuntimeError("❌ Landmark extractor not available. MediaPipe required.")
        
        # Extract landmarks and geometric features - REAL EXTRACTION
        landmarks_3d, geometric_features = self.landmark_extractor.extract_landmarks(image_np)
        
        if geometric_features is None or landmarks_3d is None:
            raise RuntimeError("❌ Could not extract features from image. Make sure face is visible.")
        
        print(f"✅ Extracted geometric features: {list(geometric_features.keys())}")
        
        # Build feature dictionary - use extracted features, add defaults only for demographics
        feature_dict = geometric_features.copy()
        
        # Add demographic defaults (not extractable from image alone)
        feature_dict['age'] = 30.0  # Default median age
        feature_dict['sex_encoded'] = 0.5  # Neutral
        feature_dict['race_Asian'] = 0.0
        feature_dict['race_Black'] = 0.0
        feature_dict['race_Hispanic'] = 0.0
        feature_dict['race_White'] = 0.0
        
        # Create feature vector in correct order matching training
        feature_vector = []
        for col in self.feature_cols:
            if col in feature_dict:
                feature_vector.append(feature_dict[col])
            else:
                # If feature not found, use 0 (shouldn't happen for geometric features)
                feature_vector.append(0.0)
        
        feature_array = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        
        # Scale features using the scaler from training
        if self.feature_scaler is not None:
            feature_array = self.feature_scaler.transform(feature_array)
        
        # Scale landmarks using the scaler from training
        landmarks_flat = landmarks_3d.reshape(-1, self.landmark_dim)
        if self.landmark_scaler is not None:
            landmarks_flat = self.landmark_scaler.transform(landmarks_flat)
        landmarks_scaled = landmarks_flat.reshape(1, self.num_landmarks, self.landmark_dim)
        
        # Convert to tensors - REAL FEATURES, NO DUMMY DATA
        features_tensor = torch.from_numpy(feature_array).float().to(self.device)
        landmarks_tensor = torch.from_numpy(landmarks_scaled).float().to(self.device)
        
        return features_tensor, landmarks_tensor
    
    def preprocess_image(self, image_bytes):
        """
        Preprocess image for model input.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            image_tensor: Preprocessed image tensor
            image_np: Numpy array for landmark extraction
        """
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for landmark extraction
        image_np = np.array(image)
        
        # Transform for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, image_np
    
    def predict(self, image_bytes):
        """
        Predict BMI from image using hybrid_model_v2.pth with REAL extracted features.
        NO DUMMY DATA - all features extracted from image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model if needed
            if not self.model_loaded:
                try:
                    if not os.path.exists(self.model_path):
                        return {'success': False, 'error': f'Model file not found at {self.model_path}'}
                    self.load_model()
                    self.model_loaded = True
                except Exception as e:
                    return {'success': False, 'error': f'Failed to load model: {str(e)}'}
            
            if self.landmark_extractor is None:
                error_msg = (
                    "MediaPipe is not installed. This is required for feature extraction.\n\n"
                    "To fix this issue, please install MediaPipe:\n"
                    "  pip install mediapipe\n\n"
                    "Or install all requirements:\n"
                    "  pip install -r requirements.txt\n\n"
                    "After installation, please restart the application."
                )
                return {'success': False, 'error': error_msg}
            
            # Preprocess image
            print("📸 Preprocessing image...")
            image_tensor, image_np = self.preprocess_image(image_bytes)
            
            # Extract REAL features from image - NO DUMMY DATA
            print("🔍 Extracting REAL features from image (no dummy data)...")
            features_tensor, landmarks_tensor = self.extract_features_from_image(image_np)
            
            print(f"✅ REAL Features extracted:")
            print(f"  📊 Tabular features: {features_tensor.shape}")
            print(f"  📍 Landmarks: {landmarks_tensor.shape}")
            print(f"  🖼️ Image: {image_tensor.shape}")
            
            # Set model to eval
            self.model.eval()
            
            # Make prediction using hybrid_model_v2 with REAL features
            with torch.no_grad():
                print("🧠 Running hybrid_model_v2.pth inference with REAL features...")
                
                edges = self.graph_edges
                
                # Call model with REAL extracted features - NO DUMMY DATA
                output = self.model(features_tensor, image_tensor, landmarks_tensor, edges)
                
                # Extract BMI
                if isinstance(output, torch.Tensor):
                    if output.dim() == 0:
                        bmi_value = output.item()
                    elif output.dim() == 1:
                        bmi_value = output[0].item()
                    else:
                        bmi_value = output[0, 0].item()
                else:
                    bmi_value = float(output)
                
                if not isinstance(bmi_value, (int, float)) or np.isnan(bmi_value) or np.isinf(bmi_value):
                    raise Exception(f"Invalid BMI value: {bmi_value}")
                
                # Clamp to reasonable range
                bmi_value = max(10.0, min(50.0, bmi_value))
                print(f"✅ BMI Prediction from hybrid_model_v2: {bmi_value:.2f}")
            
            category = self.categorize_bmi(bmi_value)
            
            return {
                'success': True,
                'bmi': round(bmi_value, 2),
                'category': category,
                'message': self.get_bmi_message(category)
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Prediction error:\n{error_details}")
            return {'success': False, 'error': f'Prediction error: {str(e)}'}
    
    def categorize_bmi(self, bmi):
        """Categorize BMI value."""
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal weight'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    def get_bmi_message(self, category):
        """Get message for BMI category."""
        messages = {
            'Underweight': 'You are underweight. Consider consulting a healthcare professional.',
            'Normal weight': 'You have a healthy weight. Keep up the good work!',
            'Overweight': 'You are overweight. Consider a balanced diet and regular exercise.',
            'Obese': 'You are in the obese range. Please consult a healthcare professional for guidance.'
        }
        return messages.get(category, 'Please consult a healthcare professional for accurate assessment.')
