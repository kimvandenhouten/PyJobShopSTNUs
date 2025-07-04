************************************************************************
file with basedata            : md359_.bas
initial value random generator: 417790909
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  154
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       40        8       40
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   8  16
   3        3          3           7   8  16
   4        3          2           5   8
   5        3          2           6   9
   6        3          1           7
   7        3          2          10  17
   8        3          2           9  11
   9        3          3          13  17  20
  10        3          2          11  20
  11        3          1          12
  12        3          1          13
  13        3          2          14  18
  14        3          2          15  19
  15        3          1          21
  16        3          3          17  18  20
  17        3          2          19  21
  18        3          2          19  21
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     2       6    5    6    8
         2     7       6    2    6    7
         3     9       6    2    5    7
  3      1     3       5    6    5    8
         2     3       8    9    4    7
         3     5       4    3    2    6
  4      1     3       9    8    4    9
         2     3      10    6    5    9
         3     3      10    6    8    8
  5      1     2       1    8    5    4
         2     8       1    7    5    2
         3     8       1    6    5    4
  6      1     4      10   10    3    2
         2     9       7    7    3    1
         3    10       5    5    3    1
  7      1     5       6    6    3    5
         2     7       5    5    3    5
         3     9       3    4    2    4
  8      1     4       8    9    9    5
         2     8       7    6    7    2
         3     9       5    6    7    2
  9      1     4       6    5    7    5
         2     6       5    3    4    4
         3     7       4    2    2    1
 10      1     3       2    9    7   10
         2     4       2    9    6    7
         3     6       2    8    3    3
 11      1     1       2    9    5    6
         2     2       2    8    4    5
         3     6       1    6    3    5
 12      1     5       4    7    9    1
         2     9       4    6    6    1
         3    10       2    6    4    1
 13      1     4       6    1    9    7
         2     8       6    1    9    6
         3     8       2    1    9    7
 14      1     7      10    3    3    6
         2     7      10    4    4    4
         3    10      10    2    3    2
 15      1     5       3    9    2    6
         2     8       3    7    1    5
         3     9       2    7    1    5
 16      1     3      10    8    7    7
         2     5       9    7    6    7
         3     9       9    7    5    7
 17      1     2      10    4   10    4
         2     6       9    3   10    3
         3     7       9    3    9    2
 18      1     3       9    2    5    6
         2     3       8    2    5    9
         3     8       6    2    5    6
 19      1     1       7    4    9    7
         2     3       6    2    8    7
         3     4       6    1    8    7
 20      1     3       6    7    9    6
         2     4       5    6    7    5
         3     8       4    4    6    5
 21      1     1       6    9    9    9
         2     5       3    7    7    8
         3     9       1    7    6    7
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   24   22  102   96
************************************************************************
DEADLINES:
jobnr.  deadline
  3       44
  4       102
  5       39
  6       46
  7       130
  10      74
  11      27
  21      133
************************************************************************
