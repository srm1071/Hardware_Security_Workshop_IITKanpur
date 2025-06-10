`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 15.03.2023 23:32:53
// Design Name: 
// Module Name: modreduce_255mult
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
module modreduce_255mult(inp,out);
input[509:0] inp;
output [254:0] out;
wire [260:0] cl,ch;
wire [260:0] w1,w2,w3;
wire [260:0] w4,w5;
wire [260:0] w6,w7;
wire c_out1,c_out2,c_out3,c_out11,c_out21,c_out31;
wire [254:0] cl1;
wire [254:0] ch1;
wire [254:0] w11,w21,w31;
wire [254:0] w41,w51;
wire [254:0] w61,w71;
wire [255:0] outprime;
wire [254:0] prime;
assign prime=255'd57896044618658097711785492504343953926634992332820282019728792003956564819949;

//write your solution here

endmodule
