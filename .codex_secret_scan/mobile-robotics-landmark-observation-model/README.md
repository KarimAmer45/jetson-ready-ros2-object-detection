# Landmark Observation Model

Range-only landmark observation likelihoods over a 2D grid map. The implementation computes single-beacon likelihood maps and joint likelihoods for multiple beacons with independent Gaussian measurement noise.

## Run

```bash
python - <<'PY'
import numpy as np
from mobile_robotics_landmark_observation_model import joint_observation_likelihood

grid = np.zeros((20, 20))
beacons = np.array([[6, 2], [8, 18], [16, 8]])
z = np.array([10, 5, 9])
sigma = np.array([1, 5, 3])

likelihood = joint_observation_likelihood(z, sigma, beacons, grid)
print(likelihood.shape, likelihood.max())
PY
```

## Likelihood output

![mobile-robotics-landmark-observation-model result screenshot](docs/results/result-screenshot.png)

Joint range-observation likelihood heatmap over a 2D grid.


## Sensor-model notes

- Range-only landmark likelihood computation over candidate robot positions.
- Combination of independent beacon observations into a joint likelihood map.
- A small probabilistic perception module that can feed localization methods.


## Follow-up validation

- The model assumes independent Gaussian range errors.
- It does not include data association or bearing measurements.
- Next steps: add multi-modal examples and integrate with a localization filter.

