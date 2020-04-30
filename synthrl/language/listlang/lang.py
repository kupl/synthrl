# grammar for ListLang
# L -> I; I; I; ... I;        # seq: T sequences
# I -> x_i <- F               # assign: for new x_i
# F -> MAP AUOP V             # map
#    | FILTER BUOP V          # filter
#    | COUNT BUOP V           # count
#    | SCANL1 ABOP V          # scanl1
#    | ZIPWITH ABOP V V       # zipwith
#    | HEAD V                 # head
#    | LAST V                 # last
#    | MINIMUM V              # minimum
#    | MAXIMUM V              # maximum
#    | REVERSE V              # reverse
#    | SORT V                 # sort
#    | SUM V                  # sum
#    | TAKE V V               # take
#    | DROP V V               # drop
#    | ACCESS V V             # access
# V -> a_1 | a_2              # inputs
#    | x_1 | x_2 | ... | x_T  # variables
# AUOP -> +1 | -1 | *2 | /2 | *(-1) | **2 | *3 | /3 | *4 | /4
# BUOP -> >0 | <0 | %2==0 | %2==1
# ABOP -> + | * | MIN | MAX