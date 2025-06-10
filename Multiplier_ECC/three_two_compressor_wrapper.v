`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.02.2023 13:38:28
// Design Name: 
// Module Name: three_two_compressor_wrapper
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


module three_two_compressor_wrapper(a,b,c,sum,carry);

parameter W=64;
input [W-1:0] a,b,c;
output [W:0] sum, carry;

genvar i;
generate for (i=0;i<W;i=i+1)
begin
three_two_compressor comp(a[i],b[i],c[i],sum[i],carry[i+1]);
end
endgenerate


assign sum[W]=0;
assign carry[0]=0;



endmodule
