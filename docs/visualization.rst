Visualization Guide
=================

This guide covers the visualization tools available for analyzing and optimizing neural circuit policies.

Wiring Visualization
----------------

WiringVisualizer
~~~~~~~~~~~~

.. code-block:: python

    from ncps.mlx.visualization import WiringVisualizer
    
    # Create visualizer
    wiring = Random(units=100, sparsity_level=0.5)
    visualizer = WiringVisualizer(wiring)

Methods:

1. **plot_wiring**
   
   Plot network topology:

   .. code-block:: python

       visualizer.plot_wiring(
           figsize=(10, 10),
           node_size=100,
           node_color='#1f77b4',
           edge_color='#aaaaaa',
           with_labels=True,
           layout='spring'  # 'spring', 'circular', 'kamada_kawai', 'shell'
       )

2. **plot_connectivity_matrix**
   
   Plot adjacency matrix:

   .. code-block:: python

       visualizer.plot_connectivity_matrix(
           figsize=(8, 8),
           cmap='viridis'
       )

3. **plot_degree_distribution**
   
   Plot in/out degree distributions:

   .. code-block:: python

       visualizer.plot_degree_distribution(
           figsize=(12, 5)
       )

4. **plot_path_lengths**
   
   Plot path length distribution:

   .. code-block:: python

       visualizer.plot_path_lengths(
           figsize=(8, 6)
       )

Performance Visualization
--------------------

PerformanceVisualizer
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ncps.mlx.visualization import PerformanceVisualizer
    
    # Create visualizer
    visualizer = PerformanceVisualizer()

Methods:

1. **add_metrics**
   
   Add performance metrics:

   .. code-block:: python

       visualizer.add_metrics(
           loss=0.5,          # Training loss
           memory=100.0,      # Memory usage (MB)
           time=0.1,          # Execution time (s)
           tflops=1.5         # Compute throughput
       )

2. **plot_metrics**
   
   Plot performance metrics:

   .. code-block:: python

       visualizer.plot_metrics(
           metrics=['loss', 'memory', 'time', 'tflops'],
           figsize=(15, 5),
           rolling_window=1  # Moving average window
       )

3. **plot_correlation**
   
   Plot correlation between metrics:

   .. code-block:: python

       visualizer.plot_correlation(
           metric1='loss',
           metric2='tflops',
           figsize=(8, 6)
       )

Profiling Visualization
-------------------

ProfileVisualizer
~~~~~~~~~~~~~

.. code-block:: python

    from ncps.mlx.visualization import ProfileVisualizer
    from ncps.mlx.advanced_profiling import MLXProfiler
    
    # Create visualizer
    profiler = MLXProfiler(model)
    visualizer = ProfileVisualizer(profiler)

Methods:

1. **plot_compute_profile**
   
   Plot compute profiling results:

   .. code-block:: python

       visualizer.plot_compute_profile(
           figsize=(12, 5)
       )

2. **plot_memory_profile**
   
   Plot memory profiling results:

   .. code-block:: python

       visualizer.plot_memory_profile(
           figsize=(12, 5)
       )

3. **plot_stream_profile**
   
   Plot stream profiling results:

   .. code-block:: python

       visualizer.plot_stream_profile(
           figsize=(12, 5)
       )

Comparative Analysis
----------------

plot_comparison
~~~~~~~~~~~~

.. code-block:: python

    from ncps.mlx.visualization import plot_comparison
    
    # Compare different configurations
    results = {
        'Config A': {
            'loss': [0.5, 0.4, 0.3],
            'tflops': 1.5,
            'memory': 100.0
        },
        'Config B': {
            'loss': [0.6, 0.5, 0.4],
            'tflops': 1.2,
            'memory': 80.0
        }
    }
    
    plot_comparison(
        results,
        metrics=['loss', 'tflops', 'memory'],
        figsize=(15, 5)
    )

Best Practices
-----------

1. **Wiring Analysis**
   - Use different layouts to understand structure
   - Analyze connectivity patterns
   - Check path distributions
   - Monitor degree distributions

2. **Performance Tracking**
   - Track multiple metrics
   - Use rolling averages for smoothing
   - Analyze correlations
   - Compare configurations

3. **Profiling Analysis**
   - Profile different batch sizes
   - Monitor memory patterns
   - Analyze stream operations
   - Look for optimization opportunities

4. **Visualization Settings**
   - Adjust figure sizes for clarity
   - Use appropriate color schemes
   - Add grid lines for readability
   - Include legends and labels

Common Patterns
------------

1. **Training Analysis**

   .. code-block:: python

       # Track training progress
       visualizer = PerformanceVisualizer()
       
       for epoch in range(num_epochs):
           # Training step
           loss = train_step()
           
           # Profile performance
           stats = profile_step()
           
           # Record metrics
           visualizer.add_metrics(
               loss=loss,
               memory=stats['memory'],
               time=stats['time'],
               tflops=stats['tflops']
           )
       
       # Plot training history
       visualizer.plot_metrics(rolling_window=10)

2. **Architecture Comparison**

   .. code-block:: python

       # Compare architectures
       results = {}
       
       for name, config in configs.items():
           # Create and train model
           model = create_model(config)
           history = train_model(model)
           
           # Store results
           results[name] = {
               'loss': history['loss'],
               'memory': profile_memory(model),
               'tflops': profile_compute(model)
           }
       
       # Plot comparison
       plot_comparison(results)

3. **Optimization Analysis**

   .. code-block:: python

       # Analyze optimization
       profiler = MLXProfiler(model)
       visualizer = ProfileVisualizer(profiler)
       
       # Profile different configurations
       for batch_size in batch_sizes:
           profiler.profile_all(batch_size=batch_size)
       
       # Plot profiles
       visualizer.plot_compute_profile()
       visualizer.plot_memory_profile()
       visualizer.plot_stream_profile()

Troubleshooting
------------

1. **Poor Visualization**
   - Adjust figure sizes
   - Change color schemes
   - Modify layouts
   - Add grid lines

2. **Memory Issues**
   - Reduce batch sizes
   - Clear previous plots
   - Use sparse matrices
   - Profile memory usage

3. **Performance Issues**
   - Optimize profiling frequency
   - Reduce visualization complexity
   - Use appropriate layouts
   - Monitor resource usage

Getting Help
----------

If you need visualization assistance:

1. Check example notebooks
2. Review documentation
3. Join community discussions
4. File issues on GitHub
