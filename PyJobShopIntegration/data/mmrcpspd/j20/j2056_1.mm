************************************************************************
file with basedata            : md376_.bas
initial value random generator: 10728
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  163
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       24        2       24
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           5   9
   3        3          3           8  20  21
   4        3          3           6  15  17
   5        3          3           7  10  15
   6        3          3           9  11  19
   7        3          3           8  11  16
   8        3          1          12
   9        3          2          10  18
  10        3          2          13  14
  11        3          2          13  14
  12        3          1          17
  13        3          1          20
  14        3          2          20  21
  15        3          2          18  19
  16        3          2          17  18
  17        3          1          19
  18        3          1          21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     4       7    3    7    7
         2     5       5    2    7    5
         3     9       4    2    6    3
  3      1     3       9    7    6    7
         2     6       6    5    3    6
         3     9       5    3    3    6
  4      1     1       6    7    5    5
         2     6       5    7    4    5
         3     7       2    4    4    5
  5      1     1       6    7    7   10
         2     7       4    6    6    9
         3     8       3    5    4    7
  6      1     7       8    7    7   10
         2     8       7    7    7    8
         3     9       4    6    7    8
  7      1     3       8    6    6    1
         2     4       8    4    6    1
         3     9       8    3    6    1
  8      1     2       9    7    6    6
         2     6       9    5    5    5
         3     9       8    3    3    4
  9      1     3      10    4    5    6
         2     8       9    4    3    5
         3    10       7    4    1    3
 10      1     3      10    8    9    6
         2     4       5    7    9    5
         3     8       1    6    8    5
 11      1     1       4    6    9    4
         2     5       3    6    9    4
         3     8       2    3    8    4
 12      1     2       5    5    8    3
         2     7       5    3    8    3
         3     9       4    2    8    2
 13      1     2       3    8   10    5
         2     3       2    8    9    4
         3     9       2    7    8    2
 14      1     1      10    3    9    5
         2     3       6    2    7    3
         3     4       5    2    3    1
 15      1     2       9   10    2    9
         2     6       8    7    1    7
         3     8       7    5    1    5
 16      1     4       6    7   10   10
         2     6       5    5    5    7
         3     7       4    2    4    6
 17      1     2       9    6    4    2
         2     4       9    6    3    2
         3     7       9    6    2    2
 18      1     1       9    8   10    7
         2     5       7    5   10    6
         3     6       7    5    9    4
 19      1     4       6    3    6    7
         2     6       3    3    5    7
         3     9       3    2    5    6
 20      1     8       4    8    3    5
         2     9       4    5    3    5
         3    10       1    2    2    5
 21      1     6       6   10    8    8
         2     8       6    8    2    6
         3     8       6    7    3    4
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   29   24  126  113
************************************************************************
DEADLINES:
jobnr.  deadline
  10      40
  19      147
************************************************************************
