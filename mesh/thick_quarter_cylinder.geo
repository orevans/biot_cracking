cl_1 = .1;
ir = .1;
Point(1) = {ir, 0, 0, cl_1};
Extrude {8.5-ir, 0, 0} {
  Point{1};
}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Line{1};
}
Physical Line(1) = {3};  // inner
Physical Line(2) = {4};  // outer
Physical Line(3) = {1};  // 3 o'clock radius (x-axis)
Physical Line(4) = {2};  // 12 o'clock radius (y-axis)
Physical Surface(1) = {5};
