`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/13/2024 08:46:45 PM
// Design Name: 
// Module Name: subtractor_256_bit_old
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


module subtractor_256_bit_old(a,b,cin,sum,cout);
parameter w=64;
input [w-1:0] a,b;
input cin;
output [w-1:0] sum;
output cout;

assign {cout,sum}=a-b-cin;

endmodule