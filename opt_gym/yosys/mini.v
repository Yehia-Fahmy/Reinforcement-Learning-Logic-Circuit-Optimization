// Example 1: A simple 2-to-1 Multiplexer
module mux2_to_1 (
    input  wire a,
    input  wire b,
    input  wire sel,
    output wire y
);
    assign y = sel ? b : a;
endmodule

// Example 2: A 3-bit Adder
module adder_3bit (
    input  wire [2:0] a,
    input  wire [2:0] b,
    output wire [2:0] sum,
    output wire       cout
);
    assign {cout, sum} = a + b;
endmodule


// Example 3: A 3-to-8 Decoder
module decoder_3_to_8 (
    input  wire [2:0] in,
    output wire [7:0] out
);
genvar i;
generate
    for (i = 0; i < 8; i = i + 1) begin : gen_decoder
        assign out[i] = (in == i);
    end
endgenerate
endmodule


// Example 4: A simple priority encoder
// This will output the binary representation of the first '1' it finds from the MSB.
module priority_encoder_4_to_2 (
    input  wire [3:0] in,
    output wire [1:0] out,
    output wire       valid
);
    always @(*) begin
        valid = |in; // Set valid if any input is high
        casez (in)
            4'b1???: out = 2'b11; // Highest priority
            4'b01??: out = 2'b10;
            4'b001?: out = 2'b01;
            4'b0001: out = 2'b00;
            default: out = 2'b00;
        endcase
    end
endmodule

// Example 5: 5-bit adder
module adder_5bit (
    input  wire [4:0] a,
    input  wire [4:0] b,
    output wire [4:0] sum,
    output wire       cout
);
    assign {cout, sum} = a + b;
endmodule
