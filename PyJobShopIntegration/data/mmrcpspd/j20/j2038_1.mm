************************************************************************
file with basedata            : md358_.bas
initial value random generator: 27514
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  158
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21       14       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   6   7
   3        3          1           8
   4        3          3           5  10  15
   5        3          2          16  17
   6        3          3          10  12  14
   7        3          2          11  18
   8        3          3           9  17  19
   9        3          1          15
  10        3          3          11  19  21
  11        3          1          20
  12        3          2          13  19
  13        3          2          16  17
  14        3          2          15  16
  15        3          1          18
  16        3          2          20  21
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
  2      1     2       5    4    6    7
         2     2       6    5    6    6
         3     3       5    4    6    4
  3      1     2       5    9    6    5
         2     3       5    9    4    2
         3     3       4    8    4    3
  4      1     1       6    6    6    4
         2     2       4    4    4    3
         3     9       3    3    4    3
  5      1     1       9    2    8    6
         2     8       8    2    8    4
         3     9       7    2    8    1
  6      1     1       7    9    3    5
         2     5       6    6    2    5
         3    10       5    6    1    4
  7      1     6       6    8   10   10
         2     9       4    7   10    5
         3    10       4    7    9    4
  8      1     2       6    9    9    9
         2     2       7    9    5    8
         3    10       5    1    2    7
  9      1     1       3    8    3    4
         2     5       2    8    2    3
         3     9       2    6    1    3
 10      1     1       6    5    7    8
         2     9       6    4    6    6
         3    10       6    1    6    6
 11      1     9       8    8    2    3
         2     9       8    9    5    2
         3     9       7    9    2    3
 12      1     4       9    8    3    6
         2     5       8    7    3    4
         3     9       8    7    2    4
 13      1     2       8    8   10    7
         2     4       7    6   10    5
         3     8       7    5    9    4
 14      1     1       7    9    7   10
         2     2       4    7    5    9
         3     3       1    5    4    9
 15      1     4       2    3    5    7
         2     5       2    3    4    4
         3     7       1    2    2    4
 16      1     3       8    7    9    8
         2     5       6    7    8    8
         3     9       6    6    6    7
 17      1     2       8   10   10    6
         2     5       7    8    8    5
         3     8       7    5    7    4
 18      1     3      10    3    7    5
         2     7       9    3    6    4
         3     8       9    1    5    3
 19      1     1       9    7    8    9
         2     2       9    6    8    8
         3     8       8    4    7    8
 20      1     4       3    4    9    7
         2     5       3    4    7    5
         3     6       2    3    5    4
 21      1     2       6    7    9    3
         2     3       6    6    6    3
         3    10       6    6    5    2
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   22   24  106   96
************************************************************************
DEADLINES:
jobnr.  deadline
  2       30
  3       50
  4       124
  5       33
  7       148
  8       48
  9       88
  12      13
  13      117
  14      78
  16      32
  17      125
  19      13
  20      19
************************************************************************
