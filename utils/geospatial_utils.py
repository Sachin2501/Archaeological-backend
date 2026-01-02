import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import rasterio
from rasterio.features import shapes
from pyproj import Transformer
import json

class GeospatialUtils:
    """Geospatial utilities for archaeological site mapping"""
    
    def __init__(self, source_crs='EPSG:4326', target_crs='EPSG:3857'):
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    def pixel_to_world(self, x: int, y: int, transform: any) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates"""
        lon = transform[0] + x * transform[1] + y * transform[2]
        lat = transform[3] + x * transform[4] + y * transform[5]
        return lon, lat
    
    def world_to_pixel(self, lon: float, lat: float, transform: any) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        # Inverse transform calculation
        det = transform[1] * transform[5] - transform[2] * transform[4]
        x = int((transform[5] * (lon - transform[0]) - transform[2] * (lat - transform[3])) / det)
        y = int((-transform[4] * (lon - transform[0]) + transform[1] * (lat - transform[3])) / det)
        return x, y
    
    def create_geojson_from_mask(self, mask: np.ndarray, transform: any, 
                                 class_id: int = 1, simplify_tolerance: float = 0.5) -> dict:
        """Convert binary mask to GeoJSON polygons"""
        # Extract shapes from mask
        results = (
            {'properties': {'class_id': class_id}, 'geometry': polygon}
            for polygon, value in shapes(mask.astype(np.uint8), mask=(mask == class_id), transform=transform)
            if value == class_id
        )
        
        geometries = list(results)
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for geom in geometries:
            feature = {
                "type": "Feature",
                "properties": geom["properties"],
                "geometry": geom["geometry"]
            }
            geojson["features"].append(feature)
        
        return geojson
    
    def calculate_polygon_area(self, polygon: Polygon, crs: str = 'EPSG:4326') -> float:
        """Calculate area of polygon in square meters"""
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=self.source_crs)
        
        # Project to projected CRS for area calculation
        if crs != self.source_crs:
            gdf = gdf.to_crs(crs)
        
        return float(gdf.geometry.area.iloc[0])
    
    def buffer_zones(self, geometry, buffer_distance: float) -> Polygon:
        """Create buffer zones around geometries"""
        return geometry.buffer(buffer_distance)
    
    def export_to_shapefile(self, geojson: dict, output_path: str):
        """Export GeoJSON to Shapefile"""
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        gdf.to_file(output_path, driver='ESRI Shapefile')
    
    def create_site_boundary(self, artifacts: list, buffer_meters: float = 100) -> dict:
        """Create site boundary from artifact locations"""
        if not artifacts:
            return None
        
        # Convert artifacts to points
        points = []
        for artifact in artifacts:
            if 'lat' in artifact and 'lng' in artifact:
                points.append(Point(artifact['lng'], artifact['lat']))
        
        if not points:
            return None
        
        # Create convex hull
        multipoint = MultiPolygon([p.buffer(0.0001) for p in points])
        hull = multipoint.convex_hull
        
        # Buffer the hull
        buffered = hull.buffer(buffer_meters / 111320)  # Approximate conversion to degrees
        
        # Create GeoJSON
        feature = {
            "type": "Feature",
            "properties": {
                "name": "Site Boundary",
                "buffer_distance_m": buffer_meters,
                "artifact_count": len(artifacts)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(buffered.exterior.coords)]
            }
        }
        
        return feature

def calculate_bbox(coordinates: list) -> dict:
    """Calculate bounding box from coordinates"""
    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]
    
    return {
        "min_lon": min(lons),
        "min_lat": min(lats),
        "max_lon": max(lons),
        "max_lat": max(lats),
        "center_lon": (min(lons) + max(lons)) / 2,
        "center_lat": (min(lats) + max(lats)) / 2
    }

if __name__ == '__main__':
    utils = GeospatialUtils()
    print("GeospatialUtils initialized successfully")