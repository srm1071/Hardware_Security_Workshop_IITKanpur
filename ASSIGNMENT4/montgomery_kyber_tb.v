`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/09/2024 05:20:38 PM
// Design Name: 
// Module Name: montgomery_kyber_tb
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
module montgomery_kyber_tb();
reg clk;
reg [11:0] a,b;
wire [11:0] out;

initial begin 
clk<=1;
    forever begin
    #1 clk <= ~clk;
    end end

montgomery_kyber montgomery(clk,a,b,out);

initial begin
    a=12'd3328;
    b=12'd3327;
end
 
endmodule
