# CI and Jetson Deployment

This repo includes a Jenkins pipeline and an Ansible deployment path for Jetson validation.

## Jenkins stages

1. Compile Python modules.
2. Run the unit tests.
3. Build the Docker image.

The Docker stage expects a Jenkins agent with Docker available. A dedicated Jetson agent can be added later for hardware-in-the-loop benchmarks.

## Jetson deployment

Copy the inventory and edit the host:

```bash
cp ansible/inventory.example.ini ansible/inventory.ini
ansible-playbook -i ansible/inventory.ini ansible/deploy_jetson.yml --check --diff
ansible-playbook -i ansible/inventory.ini ansible/deploy_jetson.yml
```

## Hardware evidence to capture

- JetPack/L4T version.
- Docker and NVIDIA runtime versions.
- Benchmark JSON for the selected backend.
- Thermal or power mode notes if benchmarks are compared.