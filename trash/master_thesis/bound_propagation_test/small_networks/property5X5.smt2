(declare-fun X_0 () Real)
(declare-fun X_1 () Real)
(declare-fun X_2 () Real)
(declare-fun X_3 () Real)
(declare-fun X_4 () Real)
(declare-fun Y_0 () Real)
(declare-fun Y_1 () Real)
(declare-fun Y_2 () Real)
(declare-fun Y_3 () Real)
(declare-fun Y_4 () Real)


(assert (<= (* 1.0 X_0) 1))
(assert (<= (* -1.0 X_0) 5))

(assert (<= (* 1.0 X_1) 3.2))
(assert (<= (* -1.0 X_1) 3))

(assert (<= (* 1.0 X_2) 1.05))
(assert (<= (* -1.0 X_2) 1))

(assert (<= (* 1.0 X_3) 1))
(assert (<= (* -1.0 X_3) 10))

(assert (<= (* 1.0 X_4) 12))
(assert (<= (* -1.0 X_4) 56))

(assert (<=  (* 1.0 Y_0)  0.0))
(assert (<=  (* 1.0 Y_1)  0.0))
(assert (<=  (* 1.0 Y_2)  0.0))
(assert (<=  (* 1.0 Y_3)  0.0))
(assert (<=  (* 1.0 Y_4)  0.0))
