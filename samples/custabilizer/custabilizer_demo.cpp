/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <vector>
#include "custabilizer.hpp"

void print_bit_table(const std::vector<uint32_t>& table, size_t num_qubits, size_t num_shots,
                     const char* table_name, const char* row_name)
{
  std::cout << table_name << " table (first 32 shots):\n";
  size_t words_per_qubit = (num_shots + 31) / 32;

  for (size_t qubit = 0; qubit < num_qubits; qubit++) {
    std::cout << "  " << row_name << " " << qubit << ": ";
    uint32_t word = table[qubit * words_per_qubit];
    for (size_t shot = 0; shot < std::min(size_t(32), num_shots); shot++) {
      std::cout << ((word >> shot) & 1);
      if ((shot + 1) % 8 == 0 && shot != 31) std::cout << " ";
    }
    std::cout << "\n";
  }
}

int main()
{
  std::cout << "cuStabilizer Demo: Pauli Frame Simulation\n";
  std::cout << "===============================================\n\n";

  size_t num_qubits = 3;
  size_t num_shots = 2048;
  size_t num_measurements = 2;
  uint64_t seed = 12345;

  std::string circuit_str =
      "Z_ERROR(0.3) 0\n"
      "H 0\n"
      "CNOT 0 1\n"
      "M 1 2\n";

  std::cout << "Circuit:\n" << circuit_str << "\n";
  std::cout << "Parameters:\n";
  std::cout << "  Qubits: " << num_qubits << "\n";
  std::cout << "  Shots: " << num_shots << "\n";
  std::cout << "  Measurements: " << num_measurements << "\n\n";

  try {
    custabilizer::Circuit circuit(circuit_str);
    std::cout << "Circuit created successfully\n";

    custabilizer::FrameSimulator simulator(num_qubits, num_shots, num_measurements);
    std::cout << "Frame simulator created successfully\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    simulator.apply_circuit(circuit.circuit, true, seed, stream);
    cudaStreamSynchronize(stream);
    std::cout << "Circuit applied successfully\n\n";

    std::vector<uint32_t> x_table = simulator.get_x_table();
    std::vector<uint32_t> z_table = simulator.get_z_table();
    std::vector<uint32_t> m_table = simulator.get_m_table();

    std::cout << "Results:\n";
    print_bit_table(x_table, num_qubits, num_shots, "X", "Qubit");
    std::cout << "\n";
    print_bit_table(z_table, num_qubits, num_shots, "Z", "Qubit");
    std::cout << "\n";
    print_bit_table(m_table, num_measurements, num_shots, "Measurement", "Measurement");

    cudaStreamDestroy(stream);

    std::cout << "\nDemo completed successfully!\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
