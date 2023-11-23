# Imports
from kelp_coverage_estimator import KelpCoverageEstimator

# Setup
path_to_image = 'images/input.jpg'

# Process
kce = KelpCoverageEstimator()

image = kce.preprocess_image(path_to_image)
# kce.save_image(image, 'test3')

coverage_map = kce.produce_coverage_map(image)
kce.plot_coverage_map(coverage_map)

# for a in attributes:
#     value = getattr(kce, a)
#     print(f"{a}: {value}")