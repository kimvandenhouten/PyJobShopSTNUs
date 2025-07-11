************************************************************************
file with basedata            : md377_.bas
initial value random generator: 381666724
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  162
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21       19       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3          15  16  21
   3        3          3           5   7  16
   4        3          2          11  18
   5        3          3           6  10  17
   6        3          2           8  18
   7        3          3           8   9  11
   8        3          2          12  21
   9        3          2          10  13
  10        3          1          21
  11        3          2          12  13
  12        3          2          19  20
  13        3          2          14  20
  14        3          1          15
  15        3          1          17
  16        3          2          17  18
  17        3          1          19
  18        3          2          19  20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     6       9    0    7    8
         2     6       4    0    7    9
         3     9       0   10    6    8
  3      1     2       8    0    6   10
         2     2       0    6    6   10
         3     5       9    0    5   10
  4      1     5       0    4    6    9
         2     6       4    0    4    6
         3     8       3    0    2    5
  5      1     2       3    0    6    7
         2     3       2    0    6    5
         3     9       2    0    2    3
  6      1     1       5    0    6   10
         2     4       5    0    5    7
         3     9       4    0    2    7
  7      1     1       0    4    7    6
         2     2       0    3    7    6
         3     8       0    3    7    4
  8      1     3       0    7    6    5
         2     5       0    7    5    4
         3     9       2    0    3    3
  9      1     2       0    6    6    8
         2     7       3    0    5    8
         3     8       0    5    5    6
 10      1     2       6    0    9    8
         2     7       6    0    6    7
         3    10       2    0    6    5
 11      1     1       0    9    7    8
         2     6       0    8    6    5
         3     9       7    0    5    1
 12      1     5      10    0    7    6
         2     6      10    0    5    4
         3     7       9    0    4    4
 13      1     2       1    0    3    7
         2     2       2    0    4    5
         3     8       0    5    2    5
 14      1     2       0    7    5    3
         2     8       0    5    5    2
         3     8       4    0    4    2
 15      1     2       0    7    7    3
         2     5       3    0    5    3
         3     5       0    6    2    3
 16      1     1       9    0    9    3
         2     6       5    0    7    2
         3     7       4    0    5    2
 17      1     5       6    0    9    5
         2     8       0    6    9    5
         3    10       5    0    8    5
 18      1     2       7    0    2    6
         2     5       0    8    2    6
         3    10       5    0    2    6
 19      1     4       0    5    8    9
         2     5       0    4    6    9
         3    10       5    0    5    8
 20      1     1       0    2    7    9
         2     2       0    1    6    6
         3     3      10    0    3    6
 21      1     4       9    0    9    3
         2     5       8    0    8    3
         3    10       0    1    8    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   13    9  133  134
************************************************************************
DEADLINES:
jobnr.  deadline
  2       17
  3       140
  4       43
  5       66
  6       137
  7       154
  8       125
  9       151
  10      72
  11      43
  12      112
  13      117
  14      117
  15      10
  16      70
  17      36
  18      36
  19      37
  20      52
  21      147
************************************************************************
