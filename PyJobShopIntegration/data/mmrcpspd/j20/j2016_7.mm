************************************************************************
file with basedata            : md336_.bas
initial value random generator: 1100738991
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  159
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       25       18       25
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           5   7  10
   3        3          3           8  16  19
   4        3          1          18
   5        3          3           6  12  14
   6        3          1          11
   7        3          1          17
   8        3          2           9  13
   9        3          3          11  14  15
  10        3          3          11  12  14
  11        3          2          17  21
  12        3          3          15  16  19
  13        3          1          15
  14        3          2          20  21
  15        3          2          17  18
  16        3          2          18  20
  17        3          1          20
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
  2      1     2       8    9    0    5
         2     4       7    8    0    5
         3     5       6    4    0    3
  3      1     1       8    6    9    0
         2     3       7    6    6    0
         3     7       5    5    0    5
  4      1     4       5    2    0    4
         2     8       3    2    2    0
         3     9       2    2    0    2
  5      1     1       6    6   10    0
         2     3       6    6    8    0
         3     5       5    3    0    7
  6      1     3       3    8    0    8
         2     4       2    7    0    6
         3    10       1    4    0    4
  7      1     4       7    9    6    0
         2     5       5    7    5    0
         3     8       5    3    4    0
  8      1     2       5    7    0    6
         2     5       3    7    0    2
         3    10       2    7    4    0
  9      1     1       2    7    0   10
         2     9       2    7    0    4
         3     9       2    7    8    0
 10      1     7      10    5   10    0
         2     7       8    6   10    0
         3     8       6    2   10    0
 11      1     5       3    8    7    0
         2     6       2    7    0    7
         3     7       2    6    0    2
 12      1     1       8    6    6    0
         2     4       8    3    0    3
         3     7       7    3    6    0
 13      1     1      10    8    0    2
         2     3       8    8    4    0
         3    10       7    6    3    0
 14      1     2       5    9   10    0
         2     3       5    9    0    5
         3     8       4    9    3    0
 15      1     1       5    7    0    7
         2     1       6    7    0    4
         3     8       1    6    0    4
 16      1     8       8    5    0    7
         2     9       4    4    7    0
         3     9       3    5    7    0
 17      1     2       6    8   10    0
         2     3       5    4    0    5
         3     7       2    3    0    3
 18      1     1       7    8    0    4
         2     3       6    6    6    0
         3     5       4    4    0    4
 19      1     7       8    4    0    6
         2     8       7    3    0    5
         3    10       4    1    0    3
 20      1     1       6    3    4    0
         2     7       5    2    1    0
         3     9       4    2    0    5
 21      1     6       7    9    0    9
         2     7       7    8    0    8
         3     8       5    7    0    5
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   37   40   59   62
************************************************************************
DEADLINES:
jobnr.  deadline
  2       138
  3       65
  4       28
  5       89
  7       95
  8       13
  9       97
  10      22
  11      36
  12      19
  13      59
  14      5
  15      59
  16      82
  17      53
  18      107
  19      55
  20      140
  21      10
************************************************************************
