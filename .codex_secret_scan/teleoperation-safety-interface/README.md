# Teleoperation Safety Interface

A GitHub-ready browser prototype for supervising remote robot teleoperation. The interface simulates live telemetry, safety interlocks, command gating, emergency stop behavior, and operator command limiting in a single static web app.

This project is a human-machine interface prototype. It is not a certified safety controller and must not be connected to a real robot without an independent, validated safety layer.

## Features

- Live simulated telemetry for latency, battery, human range, robot speed, tilt, and motor temperature.
- Safety interlock policy with warning and hard-stop states.
- Command pad with speed limiting, deadman hold, arm/disarm, and emergency stop.
- Scenario presets for nominal operation, crowded aisles, degraded communications, slope traversal, and low battery.
- Canvas-based cell view with geofence, robot pose, obstacles, human proximity, and command vector.
- Optional Gazebo Sim / ROS 2 bridge mode through rosbridge WebSocket.
- Static deployment workflow for GitHub Pages.

## Run locally

Open `index.html` directly in a browser, or serve the folder locally with Node:

```bash
node tools/server.mjs
```

Then visit `http://localhost:4173`.

If your environment has npm, `npm start` runs the same server and `npm run check` verifies the JavaScript syntax. If Python is easier in your environment, `python -m http.server 4173` also works.

## Project structure

```text
.
|-- index.html
|-- src/
|   |-- app.js
|   `-- styles.css
|-- tools/
|   `-- server.mjs
|-- docs/
|   |-- architecture.md
|   `-- safety-model.md
|-- sim/
|   |-- README.md
|   |-- gazebo/
|   `-- ros2/
|-- .github/workflows/pages.yml
|-- .gitignore
|-- LICENSE
|-- SECURITY.md
`-- README.md
```

## Upload to GitHub

```bash
git init
git add .
git commit -m "Initial teleoperation safety interface"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

After pushing to GitHub, enable GitHub Pages from Actions. The included workflow publishes the static site from the repository root.

## Simulation

Gazebo Sim is included as an optional integration under `sim/`. The recommended robotics stack is modern Gazebo Sim plus ROS 2, `ros_gz_bridge`, and `rosbridge_suite`. See `sim/README.md` for the world file, bridge topics, and run commands.

## Safety note

The browser app demonstrates operator workflow and interface behavior only. A production teleoperation stack should place real safety enforcement in robot firmware, a safety PLC, or another independent runtime that continues to operate if the browser, network, or operator workstation fails.

## Interface output

![teleoperation-safety-interface result screenshot](docs/results/result-screenshot.png)

Operator-interface state for connection, command, and safety feedback.


## Operator workflow

- A browser-based teleoperation UI focused on safety state visibility.
- Controls for emergency stop, speed limiting, and operator command feedback.
- Documentation for connecting the interface concept to ROS/Gazebo infrastructure.


## Safety caveats

- The browser UI demonstrates operator workflow; true safety enforcement must live outside the browser.
- Live robot integration depends on rosbridge and the surrounding ROS stack.
- Next steps: add a recorded interface walkthrough and hardware-disconnect failure tests.

