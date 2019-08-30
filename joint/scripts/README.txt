Scripts:

`run_joint_prep.sh`: Prepare dataests for joint training.
`run_joint_train.sh $NAME`: training the model for setting $NAME.
`run_joint_test.sh $NAME`: testing on two tasks the model for setting $NAME.
`run_joint_test_paraphrase.sh $NAME`: testing on paraphrase task the model for setting $NAME.

Here, $NAME can be either
`en_fr` for English-French in paper, or
`en_es` for English-Spanish in paper.
