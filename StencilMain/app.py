import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from sklearn.cluster import MiniBatchKMeans  # Faster than regular KMeans
import time
try:
    from numba import jit, cuda
    import cupy as cp  # GPU acceleration (optional)
    HAS_CUDA = cuda.is_available()
except Exception:
    from numba import jit
    HAS_CUDA = False
    cp = None

@jit(nopython=True)
def _fast_edge_density(edges):
    """Optimized edge density calculation using Numba"""
    return np.sum(edges) / (edges.shape[0] * edges.shape[1])

class FastRangoliProcessor:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and HAS_CUDA
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # Pre-computed kernels for faster morphological operations
        self.kernels = {
            'small': np.ones((3, 3), np.uint8),
            'medium': np.ones((5, 5), np.uint8),
            'large': np.ones((7, 7), np.uint8)
        }
        
        # Template cache for pattern matching
        self.template_cache = {}
        
    def rapid_complexity_analysis(self, image):
        """Ultra-fast complexity analysis using parallel processing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Parallel processing of different metrics
        futures = []
        
        # Submit different analysis tasks
        futures.append(self.thread_pool.submit(self._fast_edge_analysis, gray))
        futures.append(self.thread_pool.submit(self._fast_texture_analysis, gray))
        futures.append(self.thread_pool.submit(self._fast_symmetry_check, gray))
        
        # Collect results
        results = [future.result() for future in futures]
        edge_score, texture_score, symmetry_score = results
        
        # Quick complexity calculation
        complexity_score = (edge_score * 0.4 + texture_score * 0.4 + 
                          (1 - symmetry_score) * 0.2)
        
        # Determine layer count
        if complexity_score < 0.2:
            return "Simple", 2
        elif complexity_score < 0.5:
            return "Moderate", 4
        else:
            return "Complex", 6
    
    def _fast_edge_analysis(self, gray):
        """Fast edge analysis"""
        # Use smaller kernel for speed
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        return _fast_edge_density(edges)
    
    def _fast_texture_analysis(self, gray):
        """Fast texture analysis using Sobel gradients"""
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        
        # Fast magnitude calculation
        magnitude = np.sqrt(grad_x.astype(np.float32)**2 + grad_y.astype(np.float32)**2)
        return np.mean(magnitude) / 255.0
    
    def _fast_symmetry_check(self, gray):
        """Fast symmetry detection using downsampled image"""
        # Downsample for speed
        small = cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4))
        
        h, w = small.shape
        
        # Check vertical symmetry only (faster)
        left = small[:, :w//2]
        right = np.fliplr(small[:, w//2:])
        
        min_width = min(left.shape[1], right.shape[1])
        diff = np.mean(np.abs(left[:, :min_width] - right[:, :min_width]))
        
        return diff / 255.0
    
    def lightning_fast_stencil_generation(self, image, num_layers):
        """Ultra-fast stencil generation using optimized algorithms"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Fast preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use MiniBatchKMeans for speed
        if num_layers <= 4:
            layers = self._fast_threshold_layers(blurred, num_layers)
        else:
            layers = self._fast_kmeans_layers(blurred, num_layers)
        # Parallel morphological operations
        processed_layers = []
        futures = []
        for layer in layers:
            future = self.thread_pool.submit(self._fast_morphology, layer)
            futures.append(future)
        processed_layers = [future.result() for future in futures]
        return processed_layers

    def _fast_threshold_layers(self, gray, num_layers):
        """Fast multi-level thresholding"""
        layers = []
        # Pre-calculate thresholds
        min_val, max_val = gray.min(), gray.max()
        thresholds = np.linspace(min_val + 20, max_val - 20, num_layers)
        for threshold in thresholds:
            _, layer = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            layers.append(layer)
        return layers

    def _fast_kmeans_layers(self, gray, num_layers):
        """Fast K-means clustering using MiniBatchKMeans"""
        # Downsample for speed
        small_gray = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
        # Reshape for clustering
        pixels = small_gray.reshape(-1, 1)
        # Use MiniBatchKMeans for speed
        kmeans = MiniBatchKMeans(n_clusters=num_layers, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(pixels)
        # Resize back to original
        h, w = gray.shape
        small_h, small_w = small_gray.shape
        labels_resized = cv2.resize(labels.reshape(small_h, small_w), (w, h), interpolation=cv2.INTER_NEAREST)
        # Generate layers
        layers = []
        for i in range(num_layers):
            layer = (labels_resized == i).astype(np.uint8) * 255
            layers.append(layer)
        return layers

    def _fast_morphology(self, layer):
        """Fast morphological operations"""
        # Use pre-computed kernels
        kernel = self.kernels['small']
        
        # Single morphological operation for speed
        cleaned = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    def rapid_bridge_generation(self, layers):
        """Fast bridge generation using simplified connectivity"""
        processed_layers = []
        for layer in layers:
            # Fast connected components
            num_labels, labels = cv2.connectedComponents(layer)
            if num_labels > 3:  # Only if there are disconnected components
                # Simple bridge creation using convex hull
                processed_layer = self._create_simple_bridges(layer, labels, num_labels)
                processed_layers.append(processed_layer)
        else:
                processed_layers.append(layer)
        return processed_layers
    
    def _create_simple_bridges(self, layer, labels, num_labels):
        """Create simple bridges using convex hull approach"""
        # Find centroids of components
        centroids = []
        for i in range(1, num_labels):
            mask = (labels == i).astype(np.uint8)
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        
        # Create bridges between closest centroids
        if len(centroids) > 1:
            # Simple approach: connect to nearest neighbor
            for i in range(len(centroids) - 1):
                cv2.line(layer, centroids[i], centroids[i+1], 255, thickness=6)
        
        return layer
    
    def batch_process_stencils(self, image_paths, output_dir):
        """Process multiple images in parallel"""
        # Use process pool for true parallelism
        futures = []
        
        for path in image_paths:
            future = self.process_pool.submit(self._process_single_image, path, output_dir)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(None)
        
        return results
    
    def _process_single_image(self, image_path, output_dir):
        """Process single image (for multiprocessing)"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Fast processing pipeline
            complexity, num_layers = self.rapid_complexity_analysis(image)
            layers = self.lightning_fast_stencil_generation(image, num_layers)
            processed_layers = self.rapid_bridge_generation(layers)
            
            # Save results
            import os
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, base_name)
            os.makedirs(output_path, exist_ok=True)
            
            for i, layer in enumerate(processed_layers):
                cv2.imwrite(f"{output_path}/layer_{i+1:02d}.png", layer)
            
            return {
                'image_path': image_path,
                'complexity': complexity,
                'num_layers': num_layers,
                'output_path': output_path,
                'processing_time': time.time()
            }
            
        except Exception as e:
            return {'error': str(e), 'image_path': image_path}
    
    def gpu_accelerated_processing(self, image):
        """GPU-accelerated processing using CuPy (optional)"""
        if not self.use_gpu:
            return None
        
        try:
            # Convert to GPU array
            gpu_image = cp.asarray(image)
            
            # GPU-accelerated operations
            gpu_gray = cp.mean(gpu_image, axis=2)
            
            # Fast thresholding on GPU
            thresholds = cp.linspace(50, 200, 4)
            gpu_layers = []
            
            for threshold in thresholds:
                gpu_layer = cp.where(gpu_gray > threshold, 255, 0)
                gpu_layers.append(gpu_layer)
            
            # Convert back to CPU
            layers = [cp.asnumpy(layer).astype(np.uint8) for layer in gpu_layers]
            
            return layers
            
        except Exception as e:
            print(f"GPU processing failed: {e}")
            return None
    
    def benchmark_performance(self, image_path, iterations=10):
        """Benchmark processing performance"""
        image = cv2.imread(image_path)
        results = {}
        
        # Benchmark complexity analysis
        start_time = time.time()
        for _ in range(iterations):
            complexity, num_layers = self.rapid_complexity_analysis(image)
        complexity_time = (time.time() - start_time) / iterations
        
        # Benchmark stencil generation
        start_time = time.time()
        for _ in range(iterations):
            layers = self.lightning_fast_stencil_generation(image, num_layers)
        stencil_time = (time.time() - start_time) / iterations
        
        # Benchmark bridge generation
        start_time = time.time()
        for _ in range(iterations):
            processed = self.rapid_bridge_generation(layers)
        bridge_time = (time.time() - start_time) / iterations
        
        results = {
            'complexity_analysis_time': complexity_time,
            'stencil_generation_time': stencil_time,
            'bridge_generation_time': bridge_time,
            'total_time': complexity_time + stencil_time + bridge_time,
            'complexity': complexity,
            'num_layers': num_layers
        }
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

def overlay_layers(layers):
    if not layers:
        return None
    # Start with a white canvas
    final = np.ones_like(layers[0]) * 255
    for layer in layers:
        final[layer == 0] = 0
    return final

def remove_small_and_thin_features(layer, min_area=1000, min_width=8):
    # Find all contours
    contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(layer)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # Only keep if both area and width/height are above threshold
        if area > min_area and w > min_width and h > min_width:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=cv2.FILLED)
    return cleaned

# Usage example with performance monitoring
if __name__ == "__main__":
    # Initialize processor
    processor = FastRangoliProcessor(use_gpu=False)
    
    # Benchmark performance
    image_path = "Stencil/Images/input_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the file path and format.")
        exit(1)
    benchmark_results = processor.benchmark_performance(image_path)
    
    print("Performance Benchmark Results:")
    print(f"Complexity Analysis: {benchmark_results['complexity_analysis_time']:.4f}s")
    print(f"Stencil Generation: {benchmark_results['stencil_generation_time']:.4f}s")
    print(f"Bridge Generation: {benchmark_results['bridge_generation_time']:.4f}s")
    print(f"Total Processing Time: {benchmark_results['total_time']:.4f}s")
    print(f"Detected Complexity: {benchmark_results['complexity']}")
    print(f"Number of Layers: {benchmark_results['num_layers']}")
    
    # Process single image
    complexity, num_layers = processor.rapid_complexity_analysis(image)
    layers = processor.lightning_fast_stencil_generation(image, num_layers)
    final_layers = processor.rapid_bridge_generation(layers)
    
    # Save results
    import os
    os.makedirs("fast_output", exist_ok=True)
    for i, layer in enumerate(final_layers):
        layer = remove_small_and_thin_features(layer)
        cv2.imwrite(f"fast_output/layer_{i+1:02d}.png", layer)
    # Save combined final image
    final_image = overlay_layers([remove_small_and_thin_features(l) for l in final_layers])
    if final_image is not None:
        cv2.imwrite("fast_output/final_stencil.png", final_image)
        print("Saved combined final stencil as fast_output/final_stencil.png")
    
    print(f"Generated {len(final_layers)} stencil layers in fast_output/")
    
    # Cleanup
    processor.cleanup()