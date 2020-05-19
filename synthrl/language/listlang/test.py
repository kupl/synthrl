from synthrl.language.listlang.oracle import generate_program
from synthrl.language.listlang.oracle import generate_io

prog = generate_program([list, int], list, length=3)
prog.pretty_print()

io = generate_io(prog, 5)
print(io)