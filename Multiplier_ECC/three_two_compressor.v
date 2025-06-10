`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.02.2023 13:34:10
// Design Name: 
// Module Name: three_two_compressor
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


module three_two_compressor(a,b,c,sum,carry);

input a,b,c;
output sum, carry;


LUT6_2 #(
   .INIT(64'h000000E800000096) // Specify LUT Contents
) LUT6_2_inst (
   .O6(carry), // 1-bit LUT6 output
   .O5(sum), // 1-bit lower LUT5 output
   .I0(a), // 1-bit LUT input
   .I1(b), // 1-bit LUT input
   .I2(c), // 1-bit LUT input
   .I3(1'd0), // 1-bit LUT input
   .I4(1'd0), // 1-bit LUT input
   .I5(1'b1)  // 1-bit LUT input (fast MUX select only available to O6 output)
);



endmodule
