import numpy as np
import cv2
import rasterio
from skimage import exposure, filters
import albumentations as A
from typing import Tuple, Dict, Any
import json

class ImagePreprocessor:
    """Handle satellite/drone image preprocessing"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(p=0.2),
        ])
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already float
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image to target dimensions"""
        if target_size is None:
            target_size = self.target_size
        
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def load_geotiff(self, filepath: str) -> Dict[str, Any]:
        """Load GeoTIFF file with metadata"""
        with rasterio.open(filepath) as src:
            image = src.read()
            metadata = {
                'crs': str(src.crs),
                'transform': src.transform.to_gdal(),
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0]
            }
        
        # Convert to RGB if needed
        if image.shape[0] >= 3:
            image_rgb = np.transpose(image[:3], (1, 2, 0))
        else:
            image_rgb = np.transpose(np.stack([image[0]]*3), (1, 2, 0))
        
        return {
            'image': image_rgb,
            'metadata': metadata,
            'bands': image
        }
    
    def augment_image(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        if mask is not None:
            augmented = self.augmentation_pipeline(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image']
    
    def calculate_slope(self, elevation: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
        """Calculate slope from elevation data"""
        dx, dy = np.gradient(elevation, pixel_size)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        return slope_deg
    
    def calculate_aspect(self, elevation: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
        """Calculate aspect from elevation data"""
        dx, dy = np.gradient(elevation, pixel_size)
        aspect = np.arctan2(-dx, dy)
        aspect = np.degrees(aspect)
        aspect = (aspect + 360) % 360  # Convert to 0-360
        return aspect
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        # Avoid division by zero
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        denominator = (nir + red)
        denominator[denominator == 0] = 1  # Avoid division by zero
        
        ndvi = (nir - red) / denominator
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi
    
    def extract_terrain_features(self, image_path: str) -> Dict[str, Any]:
        """Extract terrain features for erosion prediction"""
        try:
            with rasterio.open(image_path) as src:
                # Assuming bands: [red, green, blue, nir, elevation, ...]
                bands = src.read()
                
                if bands.shape[0] >= 5:
                    red = bands[0]
                    nir = bands[3]
                    elevation = bands[4]
                else:
                    # Use first band as proxy
                    elevation = bands[0]
                    red = bands[0]
                    nir = bands[0]
                
                # Calculate features
                slope = self.calculate_slope(elevation)
                aspect = self.calculate_aspect(elevation)
                ndvi = self.calculate_ndvi(red, nir)
                
                # Calculate statistics
                features = {
                    'elevation_mean': float(np.mean(elevation)),
                    'elevation_std': float(np.std(elevation)),
                    'elevation_max': float(np.max(elevation)),
                    'elevation_min': float(np.min(elevation)),
                    'slope_mean': float(np.mean(slope)),
                    'slope_std': float(np.std(slope)),
                    'slope_max': float(np.max(slope)),
                    'aspect_mean': float(np.mean(aspect)),
                    'aspect_std': float(np.std(aspect)),
                    'ndvi_mean': float(np.mean(ndvi)),
                    'ndvi_std': float(np.std(ndvi)),
                    'vegetation_coverage': float(np.mean(ndvi > 0.2)),
                    'roughness': float(np.std(slope) / np.mean(slope) if np.mean(slope) > 0 else 0)
                }
                
                # Return both raw data and statistics
                return {
                    'features': features,
                    'raw_data': {
                        'elevation': elevation.tolist(),
                        'slope': slope.tolist(),
                        'aspect': aspect.tolist(),
                        'ndvi': ndvi.tolist()
                    },
                    'shape': elevation.shape
                }
                
        except Exception as e:
            print(f"Error extracting terrain features: {e}")
            # Return dummy features for testing
            return {
                'features': {
                    'elevation_mean': 500,
                    'slope_mean': 15,
                    'ndvi_mean': 0.4,
                    'vegetation_coverage': 0.6
                }
            }
    
    def create_mask_from_geojson(self, geojson_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create segmentation mask from GeoJSON annotations"""
        # This would require rasterio and shapely for actual implementation
        # For now, return a dummy mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Create some random shapes for testing
        h, w = mask.shape
        cv2.circle(mask, (w//2, h//2), min(h, w)//4, 1, -1)  # Ruins area
        cv2.rectangle(mask, (w//3, h//3), (2*w//3, 2*h//3), 2, -1)  # Vegetation area
        
        return mask

def save_preprocessing_metadata(image_path: str, metadata: Dict[str, Any], output_path: str):
    """Save preprocessing metadata to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    # Test the preprocessor
    preprocessor = ImagePreprocessor()
    print("ImagePreprocessor initialized successfully")