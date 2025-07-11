************************************************************************
file with basedata            : md378_.bas
initial value random generator: 115068913
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
    1     20      0       20        1       20
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           7   8   9
   3        3          2          19  20
   4        3          3           5   6  12
   5        3          2          11  17
   6        3          2          11  13
   7        3          2          10  12
   8        3          2          10  17
   9        3          3          10  14  19
  10        3          1          20
  11        3          3          18  19  21
  12        3          2          13  15
  13        3          2          16  21
  14        3          3          15  16  21
  15        3          1          18
  16        3          1          17
  17        3          1          18
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
  2      1     1       0    6    3    7
         2     1      10    0    3    5
         3     6       3    0    3    3
  3      1     2       0    3    2    8
         2     2       9    0    2    7
         3     4       0    3    2    5
  4      1     1       0    2    9    4
         2     5       6    0    9    4
         3    10       0    2    8    3
  5      1     3       3    0    4    3
         2     5       0    8    3    3
         3     7       3    0    3    2
  6      1     1       0    5    7    8
         2     4       0    3    4    5
         3     7       0    3    3    3
  7      1     2       0    6    7    4
         2     6       9    0    5    4
         3     9       6    0    5    1
  8      1     7       8    0    2    6
         2    10       0    4    1    5
         3    10       8    0    1    3
  9      1     4       0    7    7    7
         2     5       9    0    7    5
         3     7       8    0    7    4
 10      1     1       7    0    8    5
         2     5       6    0    7    4
         3     8       0    2    6    3
 11      1     1       0    8    6    5
         2     2       0    8    5    5
         3     6       0    7    5    4
 12      1     1       0    7    1   10
         2     5       2    0    1   10
         3     7       2    0    1    9
 13      1     3       5    0    4    6
         2     7       2    0    4    6
         3    10       0    7    3    5
 14      1     2       0    8    7    8
         2     5       7    0    5    8
         3     6       0    8    4    7
 15      1     1       8    0    6    8
         2     4       4    0    5    8
         3    10       0    6    4    7
 16      1     2       9    0    7    8
         2     4       0    8    5    8
         3     8       8    0    2    8
 17      1     1       0    3    7    8
         2     6       5    0    7    5
         3    10       2    0    7    5
 18      1     5       5    0    5    5
         2     6       0    6    4    4
         3     8       0    5    3    4
 19      1     3       0   10    9    2
         2     5       0    7    7    2
         3     6       6    0    4    2
 20      1     5       8    0    7    3
         2    10       7    0    6    3
         3    10       0    2    5    2
 21      1     2       3    0    9    4
         2     7       2    0    8    4
         3     8       0   10    8    4
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   21   20  117  119
************************************************************************
DEADLINES:
jobnr.  deadline
  2       100
  3       77
  4       106
  5       82
  6       81
  7       22
  8       29
  10      74
  11      84
  12      100
  14      140
  15      28
  16      39
  17      88
  18      129
  19      50
  20      83
  21      104
************************************************************************
