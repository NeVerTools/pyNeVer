(declare-fun X_0 () Real)
(declare-fun X_1 () Real)
(declare-fun Y_0 () Real)
(declare-fun Y_1 () Real)

(assert (<= (* 1.0 X_0) 1.3))
(assert (<= (* -1.0 X_0) 6))

(assert (<= (* 1.0 X_1) 0.03))
(assert (<= (* -1.0 X_1) 0.45))


(assert (<=  (* 1.0 Y_0)  0.0))
(assert (<=  (* 1.0 Y_1)  0.0))



