************************************************************************
file with basedata            : md350_.bas
initial value random generator: 473206387
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  161
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       28       10       28
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2          10  15
   3        3          3           9  10  16
   4        3          2           5   9
   5        3          3           6  14  16
   6        3          2           7  15
   7        3          3           8  10  12
   8        3          3          11  13  18
   9        3          3          14  19  21
  10        3          1          18
  11        3          3          17  19  20
  12        3          1          18
  13        3          2          17  19
  14        3          1          15
  15        3          1          20
  16        3          1          17
  17        3          1          21
  18        3          2          20  21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     3       6    4    8    0
         2    10       2    3    0    7
         3    10       1    4    6    0
  3      1     2       7    8    3    0
         2     6       7    6    3    0
         3     8       7    5    3    0
  4      1     2       5   10    4    0
         2     7       2    9    0    1
         3     9       1    8    0    1
  5      1     6       8    8    0    9
         2     8       8    7    7    0
         3    10       8    7    2    0
  6      1     1       7    4    0    2
         2     1       8    1    0    2
         3     1       5    1    2    0
  7      1     6       4    4    3    0
         2     7       4    2    0    5
         3     8       3    1    0    5
  8      1     1       7    9    0    5
         2     6       5    6   10    0
         3     9       3    4    0    5
  9      1     2       3    8    0    9
         2     3       3    3    5    0
         3    10       2    2    3    0
 10      1     5      10    3    4    0
         2     6       9    3    3    0
         3    10       8    2    0    2
 11      1     1       9    4    0    5
         2     8       5    3    0    5
         3     9       1    3    0    4
 12      1     1       9    3    6    0
         2     6       6    3    6    0
         3     9       4    3    4    0
 13      1     5       6    2    0    9
         2     7       3    2    3    0
         3     7       3    2    0    8
 14      1     1       6    7    0    8
         2     2       6    6    6    0
         3     8       6    4    5    0
 15      1     2       7    3    0    1
         2     4       5    3    3    0
         3     7       4    2    3    0
 16      1     1       8    8    7    0
         2     4       4    7    0    3
         3     5       3    6    0    3
 17      1     4       5    6    6    0
         2     5       3    2    5    0
         3     5       4    2    0    4
 18      1     1       6    7    0    8
         2     7       6    7    6    0
         3    10       5    7    0    5
 19      1     2       9    1    0    7
         2     8       7    1    3    0
         3    10       7    1    0    7
 20      1     5       8    9    0    7
         2     8       4    8    0    6
         3     9       4    7    0    4
 21      1     3       5    9    0   10
         2     3       7    8    7    0
         3     7       3    8    6    0
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   17   15   93  102
************************************************************************
DEADLINES:
jobnr.  deadline
  6       131
  9       19
  16      75
************************************************************************
