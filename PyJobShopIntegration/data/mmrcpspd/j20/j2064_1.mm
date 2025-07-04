************************************************************************
file with basedata            : md384_.bas
initial value random generator: 24105
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  151
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21       17       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          1           5
   3        3          3           7  15  21
   4        3          3           6   8  13
   5        3          2           7   8
   6        3          3          10  12  15
   7        3          1          11
   8        3          2           9  18
   9        3          3          10  16  17
  10        3          3          19  20  21
  11        3          2          13  19
  12        3          3          14  16  17
  13        3          1          14
  14        3          1          18
  15        3          3          17  18  19
  16        3          1          21
  17        3          1          20
  18        3          1          20
  19        3          1          22
  20        3          1          22
  21        3          1          22
  22        1          0        
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  R 2  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0    0
  2      1     1       6    6    7    8
         2     5       6    6    6    6
         3     8       6    4    3    3
  3      1     4       7    6    6    9
         2     5       5    5    4    8
         3     6       5    4    1    7
  4      1     1       6    5    6   10
         2     5       6    5    5    9
         3     6       5    4    4    9
  5      1     3      10    8   10    9
         2     5       6    8    6    8
         3     7       2    8    2    7
  6      1     2       7   10    9    5
         2     7       7    9    6    3
         3     9       6    8    5    1
  7      1     2       7    2    9    7
         2     7       7    2    9    4
         3     8       3    2    8    4
  8      1     2       7    6   10    4
         2     5       5    5    5    3
         3     8       5    4    3    1
  9      1     1       8    6    2    9
         2     3       4    6    2    9
         3     8       3    6    2    8
 10      1     4       5    7    5    4
         2     5       5    5    5    3
         3     9       5    2    4    2
 11      1     1       7    6    9    3
         2     7       5    6    9    3
         3     9       2    6    8    2
 12      1     7       9   10    4    9
         2     7       8   10    6    9
         3     9       3   10    4    8
 13      1     6       5    9    8    9
         2     7       3    6    7    9
         3     8       2    2    4    8
 14      1     1       7    8    6    9
         2     6       6    6    6    9
         3     7       4    4    3    9
 15      1     6       8    5    9    5
         2     7       8    5    7    4
         3    10       7    5    6    1
 16      1     1       8   10    8    4
         2     3       6    8    7    4
         3     4       5    6    7    4
 17      1     2       9    9    4   10
         2     8       8    7    3    9
         3     9       6    2    1    8
 18      1     1       6   10    7    6
         2     6       5    9    7    5
         3     8       4    9    4    3
 19      1     3       7    9    5    5
         2     4       6    6    5    5
         3     5       4    4    5    3
 20      1     6       7    9   10    8
         2     8       7    8    8    6
         3    10       4    8    7    6
 21      1     1       4    7    7    1
         2     2       4    6    5    1
         3     3       3    6    2    1
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   32   35  143  134
************************************************************************
DEADLINES:
jobnr.  deadline
  2       86
  6       119
  8       117
  9       123
  12      82
************************************************************************
