import numpy as np
from PIL import Image
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import sklearn for model loading (model contains RobustScaler objects)
try:
    import sklearn
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Model loading may fail if model contains sklearn objects.")


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
            # Output expects 3 scales concatenated: hidden * 3 = 384
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
            # Simulate 3 scales by using mean, max, and attention-weighted mean
            x_mean = x.mean(1)
            x_max = x.max(1)[0]
            x_attn = (x * torch.softmax(x.mean(-1, keepdim=True), dim=1)).mean(1)
            x_concat = torch.cat([x_mean, x_max, x_attn], dim=1)
            return self.output(x_concat)


class HybridModel(nn.Module):
    """
    Hybrid model architecture that matches the saved model structure.
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
        
        # Other heads (for multi-task learning, though we only use BMI)
        self.age_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 1)
        )
        self.sex_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 2)
        )
        self.cat_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 4)
        )

        # Log vars for multi-task learning
        self.log_vars = nn.Parameter(torch.zeros(4))
        
        self._init_weights()
    
    def _build_cnn_backbone(self):
        """
        Build a custom CNN backbone for image feature extraction.
        Output feature size: 512
        """
        return nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
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
        Forward pass matching the training code signature.
        
        Args:
            features: Tabular features [B, num_features]
            image: Image tensor [B, 3, H, W]
            graph_features: Landmark features [B, num_landmarks, landmark_dim] (optional)
            edges: Graph edges dictionary (optional)
        """
        bs = features.size(0)

        # Image features
        img_feat = self.img_backbone(image).view(bs, -1)
        img_feat = self.img_proj(img_feat)

        # Tabular features
        tab_feat = self.tab_net(features)

        # GCN features
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
                # Fallback if GCN fails
                print(f"Warning: GCN forward failed, using zeros: {e}")
                gcn_feat = torch.zeros(bs, 256, device=img_feat.device)
                combined = torch.cat([img_feat, tab_feat, gcn_feat], 1)
        else:
            # If no GCN, use zeros for gcn_feat dimension
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


class BMIPredictor:
    """
    BMI Predictor class that wraps your PyTorch ML model.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the BMI predictor.
        
        Args:
            model_path: Path to your trained PyTorch model file (.pth)
        """
        # Default model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'hybrid_model_v2.pth')
        
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.load_error = None
        
        # Try to load model, but don't fail if it doesn't exist
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.load_model()
            self.model_loaded = True
        except Exception as e:
            self.load_error = str(e)
            print(f"Warning: Model could not be loaded during initialization: {e}")
            print("Model will be loaded lazily when first prediction is requested.")
    
    def load_model(self):
        """
        Load the PyTorch ML model.
        """
        try:
            print(f"Loading model from: {self.model_path}")
            print(f"Using device: {self.device}")
            
            # Load the model
            # weights_only=False is needed because the model contains sklearn objects (RobustScaler)
            # This is safe since it's your own trained model
            loaded_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Handle different model save formats
            if isinstance(loaded_data, dict):
                # Check if it contains model_state (state_dict format)
                if 'model_state' in loaded_data:
                    # Extract model parameters
                    model_state = loaded_data['model_state']
                    num_landmarks = loaded_data.get('num_landmarks', 21)
                    landmark_dim = loaded_data.get('landmark_dim', 3)
                    num_features = len(loaded_data.get('feature_cols', [])) if loaded_data.get('feature_cols') else 36
                    
                    # Create model instance with correct signature
                    self.model = HybridModel(
                        num_features=num_features,
                        num_landmarks=num_landmarks,
                        landmark_dim=landmark_dim,
                        dropout=0.3,
                        use_gcn=True
                    )
                    
                    # Load state dict (with strict=False to handle architecture differences)
                    try:
                        self.model.load_state_dict(model_state, strict=False)
                        print("Model state_dict loaded (some layers may not match exactly)")
                    except Exception as e:
                        print(f"Warning: Could not load state_dict strictly: {e}")
                        # Try to load matching layers only
                        model_dict = self.model.state_dict()
                        pretrained_dict = {k: v for k, v in model_state.items() if k in model_dict and model_dict[k].shape == v.shape}
                        model_dict.update(pretrained_dict)
                        self.model.load_state_dict(model_dict)
                        print(f"Loaded {len(pretrained_dict)} matching layers")
                    
                    # Store scalers and other metadata
                    self.feature_scaler = loaded_data.get('feature_scaler')
                    self.landmark_scaler = loaded_data.get('landmark_scaler')
                    self.feature_cols = loaded_data.get('feature_cols')
                    self.graph_edges = loaded_data.get('graph_edges')
                    self.num_landmarks = num_landmarks
                    self.landmark_dim = landmark_dim
                    
                elif 'model' in loaded_data:
                    self.model = loaded_data['model']
                elif 'state_dict' in loaded_data:
                    # If only state_dict is saved, create model and load it
                    num_features = len(loaded_data.get('feature_cols', [])) if loaded_data.get('feature_cols') else 36
                    self.model = HybridModel(num_features=num_features)
                    self.model.load_state_dict(loaded_data['state_dict'], strict=False)
                else:
                    # Try to find a model-like object in the dict
                    # Common pattern: the model might be the whole dict or a specific key
                    self.model = loaded_data
            else:
                # If it's directly the model
                self.model = loaded_data
            
            # Set model to evaluation mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Move model to appropriate device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image_bytes):
        """
        Preprocess the uploaded image for PyTorch model input.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed image tensor ready for model input
        """
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms - adjust these based on your model's training preprocessing
        # Common preprocessing for PyTorch models
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size if your model uses different input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def predict(self, image_bytes):
        """
        Predict BMI from image using the loaded PyTorch model.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Try to load model if it wasn't loaded during initialization
            if not self.model_loaded:
                try:
                    if not os.path.exists(self.model_path):
                        return {
                            'success': False,
                            'error': f'Model file not found at {self.model_path}. Please ensure the model file exists.'
                        }
                    self.load_model()
                    self.model_loaded = True
                    self.load_error = None
                except Exception as e:
                    self.load_error = str(e)
                    import traceback
                    traceback.print_exc()
                    return {
                        'success': False,
                        'error': f'Failed to load model: {str(e)}. Please check the model file and dependencies.'
                    }
            
            if self.model is None:
                error_msg = self.load_error or 'Model not loaded. Please check model file.'
                return {
                    'success': False,
                    'error': error_msg
                }
            
            # Preprocess image
            input_tensor = self.preprocess_image(image_bytes)
            
            # Make prediction
            with torch.no_grad():
                # Call the model with correct signature: (features, image, graph_features, edges)
                # For inference, we only have images, so create dummy features
                batch_size = input_tensor.shape[0]
                num_features = len(self.feature_cols) if hasattr(self, 'feature_cols') and self.feature_cols else 36
                
                # Create dummy tabular features (zeros)
                dummy_features = torch.zeros(batch_size, num_features, device=input_tensor.device, dtype=torch.float32)
                
                # Create dummy graph features if needed
                if hasattr(self, 'num_landmarks') and hasattr(self, 'landmark_dim'):
                    num_landmarks = self.num_landmarks
                    landmark_dim = self.landmark_dim
                    dummy_graph = torch.zeros(batch_size, num_landmarks, landmark_dim, device=input_tensor.device, dtype=torch.float32)
                else:
                    dummy_graph = None
                
                # Get graph edges if available
                edges = getattr(self, 'graph_edges', None)
                
                if isinstance(self.model, nn.Module):
                    # Standard PyTorch model call with correct signature
                    output = self.model(dummy_features, input_tensor, dummy_graph, edges)
                elif callable(self.model):
                    # Fallback for other callable models
                    output = self.model(input_tensor)
                else:
                    raise Exception("Model is not callable. Please check model structure.")
                
                # Handle different output formats
                if isinstance(output, torch.Tensor):
                    # If output is a tensor, extract the value
                    if output.dim() == 0:
                        # Scalar tensor
                        bmi_value = output.item()
                    elif output.dim() == 1:
                        # 1D tensor, take first element
                        bmi_value = output[0].item()
                    elif output.dim() == 2:
                        # 2D tensor (batch, features), take first element
                        bmi_value = output[0][0].item() if output.shape[1] > 1 else output[0].item()
                    else:
                        # Higher dimensions, flatten and take first
                        bmi_value = output.flatten()[0].item()
                elif isinstance(output, (list, tuple)):
                    # If output is a list/tuple, take first element
                    bmi_value = float(output[0])
                elif isinstance(output, dict):
                    # If output is a dict, look for common keys
                    if 'bmi' in output:
                        bmi_value = float(output['bmi'])
                    elif 'prediction' in output:
                        bmi_value = float(output['prediction'])
                    else:
                        # Take first value
                        bmi_value = float(list(output.values())[0])
                else:
                    # Try to convert to float
                    bmi_value = float(output)
            
            # Ensure BMI is in a reasonable range (adjust if needed)
            bmi_value = max(10.0, min(50.0, bmi_value))  # Clamp between 10 and 50
            
            # Categorize BMI
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
            print(f"Prediction error: {error_details}")
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }
    
    def categorize_bmi(self, bmi):
        """
        Categorize BMI value into standard categories.
        
        Args:
            bmi: BMI value
            
        Returns:
            BMI category string
        """
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal weight'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    def get_bmi_message(self, category):
        """
        Get a message based on BMI category.
        
        Args:
            category: BMI category string
            
        Returns:
            Informative message
        """
        messages = {
            'Underweight': 'You are underweight. Consider consulting a healthcare professional.',
            'Normal weight': 'You have a healthy weight. Keep up the good work!',
            'Overweight': 'You are overweight. Consider a balanced diet and regular exercise.',
            'Obese': 'You are in the obese range. Please consult a healthcare professional for guidance.'
        }
        return messages.get(category, 'Please consult a healthcare professional for accurate assessment.')
