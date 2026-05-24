# Monte Carlo Localization

Particle-filter localization on a 2D occupancy map using odometry and laser observations. The project includes scalar and vectorized implementations for motion sampling, beam likelihood weighting, and low-variance resampling.

## Run

```bash
python - <<'PY'
import pickle
import numpy as np
import mobile_robotics_monte_carlo_localization as monte_carlo_localization

data = pickle.load(open("dataset_mit_csail.p", "rb"))
map_res = 0.1
num_particles = 200
particles = monte_carlo_localization.init_uniform(num_particles, data["img_map"], map_res)
gridmap = 255 - data["img_map"]

particles = monte_carlo_localization.mc_localization(
    data["odom"][:10],
    data["z"][:10],
    num_particles,
    particles,
    [0.1, 0.1, 0.1, 0.1],
    gridmap,
    data["likelihood_map"],
    map_res,
    data["img_map"],
    parallel_mode=True,
    surpress_pb=True,
)
print(particles.shape)
PY
```

## Particle-filter output

![mobile-robotics-monte-carlo-localization result screenshot](docs/results/result-screenshot.png)

Particle cloud after odometry and laser updates on the bundled map.


## MCL workflow

- Particle-filter localization with motion sampling, beam weighting, and resampling.
- Vectorized/parallel code paths for practical performance experiments.
- Use of a real map-like occupancy image and laser likelihood map.


## Dataset notes

- The README example runs only a short slice of the dataset for quick inspection.
- Particle count and likelihood parameters strongly affect convergence.
- Next steps: add trajectory error metrics and a saved animation of resampling behavior.

