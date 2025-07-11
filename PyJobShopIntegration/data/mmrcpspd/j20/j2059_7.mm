************************************************************************
file with basedata            : md379_.bas
initial value random generator: 1934606270
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  157
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       20       19       20
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   8  11
   3        3          2          14  17
   4        3          3          10  11  12
   5        3          3           6  13  20
   6        3          3           7   9  10
   7        3          3          12  14  15
   8        3          3          13  14  18
   9        3          1          15
  10        3          1          16
  11        3          3          15  20  21
  12        3          2          16  17
  13        3          1          16
  14        3          1          19
  15        3          1          19
  16        3          1          21
  17        3          1          18
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
  2      1     1       0    8    7    7
         2     8       7    0    6    5
         3    10       7    0    5    3
  3      1     2       8    0    4   10
         2     8       0    3    3   10
         3     8       2    0    2   10
  4      1     3       0    8    5   10
         2     4       0    7    5    9
         3     6       0    7    4    7
  5      1     4       0    7    7    3
         2     9       7    0    3    2
         3     9       0    2    1    2
  6      1     6       0    7    6    7
         2     9       9    0    4    4
         3    10       0    6    2    4
  7      1     1       0    3    7   10
         2     5       0    3    7    9
         3     6       0    3    7    7
  8      1     2       9    0    6    9
         2     6       8    0    5    6
         3     9       0    1    5    4
  9      1     1       4    0    6    6
         2     9       0    4    4    5
         3     9       3    0    5    5
 10      1     1       8    0    7   10
         2     1       0    5    7   10
         3     4       0    1    6    5
 11      1     8       0    6    7    6
         2     9       7    0    5    5
         3    10       3    0    4    1
 12      1     1       0    4    9    8
         2     6       0    3    6    8
         3     7       0    2    3    7
 13      1     3       7    0    5    9
         2    10       4    0    5    9
         3    10       0    7    3    8
 14      1     3       6    0    2    8
         2     7       0    4    1    4
         3    10       3    0    1    3
 15      1     6      10    0    6   10
         2     7       0    9    4    7
         3     9       8    0    4    6
 16      1     1       0   10    4    9
         2     2       7    0    3    8
         3     4       0    6    1    8
 17      1     4       0    7    7   10
         2     4      10    0    8   10
         3     5       5    0    7   10
 18      1     2       0    9    4    4
         2     2       2    0    4    4
         3     9       0    9    1    4
 19      1     1       0    9    8    2
         2     7       0    8    6    2
         3    10       0    7    5    2
 20      1     4       3    0    8    5
         2     5       0    6    8    5
         3     7       0    5    5    4
 21      1     1       0    5    5    6
         2     4       0    3    4    4
         3     5       0    1    3    3
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   23  121  149
************************************************************************
DEADLINES:
jobnr.  deadline
  2       135
  3       89
  4       15
  5       152
  6       109
  7       4
  8       29
  9       149
  10      90
  11      82
  12      83
  13      37
  14      108
  15      145
  16      117
  17      97
  18      51
  19      151
  20      43
  21      98
************************************************************************
