`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/09/2024 04:34:57 PM
// Design Name: 
// Module Name: modular_adder_kyber_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module modular_adder_kyber_tb();
reg [11:0] inp1,inp2;
wire [11:0] out;

modular_adder_kyber adder(inp1,inp2,out);

initial begin
    inp1=12'd3327;
    inp2=12'd10;
end

endmodule
