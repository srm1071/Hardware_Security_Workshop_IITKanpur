`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12.12.2024 00:23:38
// Design Name: 
// Module Name: test_adder
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


module test_adder(

    );
    
parameter W=256;
reg [W-1:0] a,b;
reg clk,rst;
wire [W-1:0] out;

initial begin 
    clk=1'b0;
    forever begin
    #10 clk = ~clk;
    end end


//carry_chain_sum_new_for_256 adder(mode,a,b,sum,c_out);

simple_modular_adder adder(a,b,clk,rst,out,done);

//carry64lookup lookahead(a,b,c_in,sum,c_out1,g,p);

initial begin
    //c_in=1'b1;
    rst=1'b1; 
//    a=64'd18446744073709551606;
//    b=64'd18446744073709551604;
    a=256'h7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec;
    b=256'h7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec;
    #100 rst=1'b0;
    // mod output: 6322e973ec7ee058b8cc396fde97386591f02e41fb01fcc99663861135d0c41c
end
endmodule
