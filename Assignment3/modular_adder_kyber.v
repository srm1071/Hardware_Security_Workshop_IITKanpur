`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 24.04.2023 11:37:55
// Design Name: 
// Module Name: bfu_adder
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
module modular_adder_kyber(inp1,inp2,out);
input [11:0] inp1,inp2;
output [12:0] out;
wire [12:0] w1,w3;

assign w1=inp1+inp2;
assign w3=inp1+inp2-12'd3329;

assign out=(w3[12])?w1[11:0]:w3[11:0];

endmodule
