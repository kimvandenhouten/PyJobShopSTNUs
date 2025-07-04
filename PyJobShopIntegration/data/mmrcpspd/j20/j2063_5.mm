************************************************************************
file with basedata            : md383_.bas
initial value random generator: 1600514532
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  178
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       35       12       35
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          14  17  20
   3        3          3           8  15  18
   4        3          2           5   8
   5        3          3           6   7  10
   6        3          2           9  20
   7        3          3           9  15  18
   8        3          3           9  16  19
   9        3          1          21
  10        3          3          11  18  20
  11        3          1          12
  12        3          2          13  19
  13        3          3          14  15  16
  14        3          1          21
  15        3          1          17
  16        3          1          17
  17        3          1          21
  18        3          1          19
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       9    7    7    3
         2     7       8    7    7    2
         3    10       4    7    7    2
  3      1     1       9    4    4    4
         2     4       5    2    3    3
         3     7       4    1    3    2
  4      1     1       4    8    4    5
         2     4       2    7    3    4
         3    10       2    7    1    4
  5      1     1       7    4    3    5
         2     4       7    3    2    5
         3     5       4    3    2    5
  6      1     8       3    9    8    9
         2     9       2    9    6    4
         3    10       1    7    2    4
  7      1     2       8    8    9    3
         2    10       3    7    8    3
         3    10       3    6    9    3
  8      1     4       6    4    7    7
         2     8       4    4    6    7
         3    10       3    4    2    7
  9      1     5      10    8    3   10
         2     9       7    7    2    6
         3    10       4    5    1    2
 10      1     4      10    6    9   10
         2     6       8    5    3    7
         3     6       9    5    4    6
 11      1     7       8   10    2   10
         2     8       8    9    2    8
         3    10       7    9    1    8
 12      1     7       5   10    7    5
         2     9       4    9    7    4
         3     9       5    9    6    3
 13      1     3       8   10    9    4
         2     5       3   10    9    3
         3     9       1   10    8    1
 14      1     4       9   10    8    5
         2     6       9    9    8    4
         3    10       8    9    8    3
 15      1     4       3   10    8    9
         2     8       2    6    7    4
         3     8       3    5    8    3
 16      1     2       9    5    9    7
         2     6       5    5    8    7
         3     8       2    4    7    7
 17      1     6      10    9    3    5
         2     8       8    7    2    4
         3    10       7    5    2    4
 18      1     3       9    4    7    8
         2     5       9    2    6    6
         3    10       8    1    5    5
 19      1     2       9    3    3    7
         2     5       7    3    2    7
         3     8       7    2    2    7
 20      1     5       5    5    4   10
         2     7       4    3    3    9
         3     9       4    1    3    9
 21      1     2       6    6    6    9
         2     8       5    6    5    9
         3     9       3    3    5    8
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   23  120  135
************************************************************************
DEADLINES:
jobnr.  deadline
  2       165
  3       85
  4       71
  6       108
  7       176
  8       43
  9       162
  10      5
  11      59
  12      49
  18      120
  20      19
************************************************************************
