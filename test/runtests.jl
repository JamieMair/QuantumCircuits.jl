using TestItemRunner

using QuantumCircuits

QuantumCircuits.install_mps_support()
QuantumCircuits.init_mps_support()

@run_package_tests
