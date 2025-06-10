
1.Install icarus-verilog for windows (http://bleyer.org/icarus/)

2.add path to environment variables

3.Open with notepad and create a "main.v" and a "main_tb.v" file in the (c:\iverilog\bin) folder. [main.v and main_tb.v is attached]

4.open command promt and goto " cd c:\iverilog\bin "

5.create a folder in the cmd, keep both main.v and main_tb.v in the same folder

5.open command promt open the same directory and type - " iverilog -o circuit main.v main_tb.v " in command prompt

6.a file will be created named 'circuit' in the same folder where main.v and main_tb.v existis

7.type " vvp circuit " in cmd

8.a " .vcd " file will be created

9.open that file using "GTKwave"

10.Within gtkwave append each input and outputs
