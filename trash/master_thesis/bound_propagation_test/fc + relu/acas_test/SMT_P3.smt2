(declare-fun X_0 () Real)
(declare-fun X_1 () Real)
(declare-fun X_2 () Real)
(declare-fun X_3 () Real)
(declare-fun X_4 () Real)
(declare-fun FC6_0 () Real)
(declare-fun FC6_1 () Real)
(declare-fun FC6_2 () Real)
(declare-fun FC6_3 () Real)
(declare-fun FC6_4 () Real)

(assert (<= (* -1.0 X_0) 0.30353115613746867))
(assert (<= (* 1.0 X_0) -0.29855281193475053))

(assert (<= (* -1.0 X_1) 0.009549296585513092))
(assert (<= (* 1.0 X_1) 0.009549296585513092))

(assert (<= (* -1.0 X_2) -0.4933803235848431))
(assert (<= (* 1.0 X_2) 0.4997465213085185))

(assert (<= (* -1.0 X_3) -0.3))
(assert (<= (* 1.0 X_3) 0.5))
(assert (<= (* -1.0 X_4) -0.3))
(assert (<= (* 1.0 X_4) 0.3333333333333333))

(assert (<= (+ (* 1.0 FC6_0) (* -1.0 FC6_1)) 0.0))
(assert (<= (+ (* 1.0 FC6_0) (* -1.0 FC6_2)) 0.0))
(assert (<= (+ (* 1.0 FC6_0) (* -1.0 FC6_3)) 0.0))
(assert (<= (+ (* 1.0 FC6_0) (* -1.0 FC6_4)) 0.0))
