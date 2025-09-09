#!/bin/bash

yosys -p "read_verilog mini.v; synth -top mux2_to_1; rename -top top; flatten; abc9 -lut 2; clean; write_blif mux2_to_1.blif"
yosys -p "read_verilog mini.v; synth -top adder_3bit; rename -top top; flatten; abc9 -lut 2; clean; write_blif adder_3bit.blif"
yosys -p "read_verilog mini.v; synth -top decoder_3_to_8; rename -top top; flatten; abc9 -lut 2; clean; write_blif decoder_3_to_8.blif"
yosys -p "read_verilog mini.v; synth -top priority_encoder_4_to_2; rename -top top; flatten; abc9 -lut 2; clean; write_blif priority_encoder_4_to_2.blif"
yosys -p "read_verilog mini.v; synth -top adder_5bit; rename -top top; flatten; abc9 -lut 2; clean; write_blif adder_5bit.blif"
