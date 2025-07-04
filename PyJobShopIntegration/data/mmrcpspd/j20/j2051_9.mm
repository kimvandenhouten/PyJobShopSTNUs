************************************************************************
file with basedata            : md371_.bas
initial value random generator: 2073504841
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  174
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       22        9       22
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          3           6   9  14
   3        3          3           5   7  15
   4        3          3           6  10  12
   5        3          2          16  18
   6        3          1           8
   7        3          3          13  14  16
   8        3          1          17
   9        3          3          13  19  20
  10        3          3          11  14  16
  11        3          2          17  18
  12        3          1          15
  13        3          1          21
  14        3          2          18  19
  15        3          2          17  20
  16        3          1          21
  17        3          2          19  21
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
  2      1     6       7    0    9   10
         2     9       7    0    8   10
         3    10       0    8    8   10
  3      1     6       0    8    5    7
         2     9       6    0    4    6
         3    10       6    0    4    5
  4      1     2       0    9    5   10
         2     4       5    0    5   10
         3    10       5    0    4    9
  5      1     1       4    0    6    9
         2     1       0    9    5    7
         3     8       0    4    5    4
  6      1     1       0    9    6    9
         2     2       0    5    5    9
         3     8       5    0    5    9
  7      1     1       0    7    2    6
         2     3       0    3    2    4
         3    10       0    2    1    3
  8      1     1       8    0    3    5
         2     9       0    6    3    4
         3    10       0    4    3    2
  9      1     2       4    0    6    7
         2     5       0    4    5    5
         3     9       4    0    5    3
 10      1     3       0    3    6    6
         2     4       0    3    5    6
         3    10       0    3    3    4
 11      1     2       7    0    7    2
         2     8       0    2    5    2
         3     9       7    0    4    1
 12      1     3       2    0    7    5
         2     3       0    8    6    4
         3    10       0    4    4    3
 13      1     2       4    0    7    4
         2     3       0    6    6    3
         3     5       3    0    5    2
 14      1     2       7    0    5    6
         2     8       0    6    4    5
         3    10       0    4    3    2
 15      1     4       9    0    7   10
         2     4       9    0    9    8
         3     7       0    6    3    7
 16      1     7       0    7   10    8
         2     7       8    0    9   10
         3    10       8    0    9    5
 17      1     5       1    0    7    9
         2     5       0    6    5    7
         3     8       0    6    2    6
 18      1     3       7    0    7    6
         2     6       0    3    7    5
         3     8       2    0    7    4
 19      1     5       8    0    8    9
         2     6       7    0    7    7
         3     7       0    9    4    7
 20      1     2       0    8   10    7
         2     3       0    6    8    6
         3     6       4    0    7    3
 21      1     7      10    0   10    9
         2     9       0    3    7    9
         3     9       6    0    7    8
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   25   26  125  134
************************************************************************
DEADLINES:
jobnr.  deadline
  2       11
  4       88
  6       53
  7       93
  8       31
  10      55
  11      30
  12      16
  13      136
  15      108
  16      98
  17      159
  18      151
  19      30
  20      74
  21      140
************************************************************************
