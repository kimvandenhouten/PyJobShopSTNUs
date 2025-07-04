************************************************************************
file with basedata            : md332_.bas
initial value random generator: 685790453
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  22
horizon                       :  144
RESOURCES
  - renewable                 :  2   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1     20      0       21        3       21
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        3          2           9  16
   3        3          3           5   7   8
   4        3          3          15  18  21
   5        3          2           6  10
   6        3          3          12  13  14
   7        3          1          11
   8        3          3           9  10  19
   9        3          1          14
  10        3          3          12  16  18
  11        3          3          12  14  16
  12        3          1          15
  13        3          3          17  19  21
  14        3          2          17  21
  15        3          1          20
  16        3          1          20
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
  2      1     6       0    5    0    7
         2     8       0    5    0    6
         3     9       7    0    0    6
  3      1     1       0    6    0    5
         2     2       0    4    7    0
         3     4       0    1    4    0
  4      1     2      10    0    0    3
         2     6       8    0    7    0
         3     7       0    3    0    3
  5      1     4       8    0    0    7
         2     6       6    0    8    0
         3     6       0    8    0    5
  6      1     1       0    6    0    9
         2     2       5    0    0    5
         3     3       4    0    0    4
  7      1     4       8    0    9    0
         2     4       0    4    0    7
         3     5       8    0    0    6
  8      1     4       2    0    0    3
         2     5       0    8    7    0
         3    10       0    6    0    2
  9      1     1       5    0    6    0
         2     7       0    6    0   10
         3     9       5    0    3    0
 10      1     3       8    0    6    0
         2     5       4    0    0    5
         3     9       4    0    0    4
 11      1     4       0    6    0    8
         2     6       6    0    7    0
         3     6       0    5    0    5
 12      1     5       3    0    1    0
         2     6       0    9    0    8
         3     7       3    0    0    4
 13      1     6       3    0    0    9
         2     6       4    0    5    0
         3     6       7    0    0    7
 14      1     4       0    1    0    6
         2     5       0    1    3    0
         3     5       4    0    0    4
 15      1     1       0    4    0    8
         2     6       0    4    0    4
         3     7       0    1    7    0
 16      1     3       0    6    9    0
         2     6       9    0    7    0
         3    10       9    0    0    6
 17      1     1       0    8    0    5
         2     4       0    2    0    3
         3     7       3    0    0    3
 18      1     4       7    0    0    6
         2     5       0    6    0    3
         3     8       0    3    5    0
 19      1     7       0    6    0   10
         2     8       0    4    5    0
         3     9       0    3    4    0
 20      1     1       7    0    0    3
         2     4       0    9    0    3
         3     8       0    7    0    3
 21      1     8       0    9    5    0
         2     9       0    9    4    0
         3     9       4    0    0    7
 22      1     0       0    0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  R 2  N 1  N 2
   35   30   49   74
************************************************************************
DEADLINES:
jobnr.  deadline
  2       119
  4       40
  6       47
  7       68
  8       49
  9       98
  11      61
  13      14
  15      18
  16      134
  17      7
  18      32
  19      69
  20      31
  21      81
************************************************************************
