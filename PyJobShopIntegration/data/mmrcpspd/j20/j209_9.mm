************************************************************************
file with basedata            : md329_.bas
initial value random generator: 1844828534
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  149
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21       16       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          15  16  17
   3        3          2           6  10
   4        3          3           5  12  14
   5        3          3           7   8   9
   6        3          3           9  11  14
   7        3          2          10  19
   8        3          2          10  19
   9        3          1          15
  10        3          2          11  17
  11        3          2          13  16
  12        3          2          20  21
  13        3          1          15
  14        3          3          19  20  21
  15        3          1          18
  16        3          1          18
  17        3          1          18
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
  2      1     5       3    0    0    5
         2     6       2    0    6    0
         3    10       0    1    5    0
  3      1     3       7    0    7    0
         2     3       0    3    7    0
         3    10       0    2    6    0
  4      1     2       9    0    0    4
         2     4       9    0    4    0
         3     6       6    0    4    0
  5      1     1       3    0    7    0
         2     2       0    2    0    4
         3     3       3    0    1    0
  6      1     2      10    0    0   10
         2     3       5    0    4    0
         3     6       5    0    3    0
  7      1     5       0    5    7    0
         2     6       0    2    0   10
         3     7       9    0    5    0
  8      1     2       8    0    0    7
         2     3       6    0    6    0
         3     6       0    4    0    7
  9      1     3       5    0    0    4
         2     8       2    0    0    2
         3     9       0    9    6    0
 10      1     1       5    0    4    0
         2     2       0    4    0    2
         3     4       0    2    3    0
 11      1     2       0    4    0    3
         2     7       0    4    0    1
         3     9       0    4    9    0
 12      1     1       7    0    4    0
         2     2       7    0    2    0
         3     8       0    7    0    7
 13      1     2       2    0   10    0
         2     7       2    0    6    0
         3    10       2    0    0    8
 14      1     2       0    8    9    0
         2     5       6    0    0    9
         3     5       8    0    6    0
 15      1     2       8    0   10    0
         2     3       8    0    0    6
         3     7       7    0    0    5
 16      1     1       0    2    0    8
         2     2       7    0    0    8
         3     6       0    2    0    6
 17      1     4       0    5   10    0
         2     5       4    0    9    0
         3     9       0    4    0    8
 18      1     5       9    0    0    4
         2     8       0    9    4    0
         3     9       0    6    4    0
 19      1     3       0    9    9    0
         2     4       0    7    9    0
         3     8       0    6    0    4
 20      1     1       0   10    6    0
         2     5       0    8    0   10
         3     9       0    4    0    9
 21      1     1       3    0    4    0
         2     3       0    9    1    0
         3     8       0    9    0    7
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13   13   66   63
************************************************************************
DEADLINES:
jobnr.  deadline
  5       59
  7       68
  8       135
  14      86
  18      118
************************************************************************
