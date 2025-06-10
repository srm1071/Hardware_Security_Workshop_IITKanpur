`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11.12.2024 23:32:42
// Design Name: 
// Module Name: simple_modular_adder
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


module simple_modular_adder(a,b,clk,rst,out,done);


input [255:0] a,b;
input clk,rst;
output [255:0] out;
wire [255:0] prime;
wire [255:0] add_out,sub_out;
output done;
wire sign;

assign prime=256'h7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed;

//choose nearest eighth and third multiple of (WB+4)*16

reg [255:0] reg_a, reg_b;

wire [63:0] sum_3,sum_4;
wire cout_3,cout_4;
reg cin_3,cin_4,rst_r,rst_rr;

reg [255:0] reg_sum_3,reg_sum_4,reg_p;
wire sum_4_en,sum_3_en;
reg [4:0] count;


wire done;

//write your code here




adder_189 #(.W(64)) add2(reg_a[63:0],reg_b[63:0],cin_3,sum_3,cout_3);

subtractor_256_bit_old #(.W(64)) sub(reg_sum_3[255:192],reg_p[63:0],cin_4,sum_4,cout_4);



endmodule




