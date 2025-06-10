`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.02.2023 13:19:17
// Design Name: 
// Module Name: register_file
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


module register_file(in, clk,rst,sel,out);

input [16:0] in;
input clk,rst,sel;
output reg [237:0] out;



always@(posedge clk)
begin
if(rst)
out<=238'h0;
else if(sel)
out<={in,out[237:17]};
else
out<=out;
end





endmodule
