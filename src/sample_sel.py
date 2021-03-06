"""
#############################################################################################################

selection of samples taken manually from all SYNTHIA sequences, and classified by type of event
this dictionary of samples is imported by exec_dset.py and transformed into the standard string
format used there by exec_dset.sel_test()

this selection is then used by tester.accur_sel_time(), in the format converted by exec_dset.sel_test()

    Alice   2019

#############################################################################################################
"""

sample_sel = {
    "SYNTHIA-SEQS-01-DAWN": (
        ( "000324", "taking over car on the right" ),
        ( "000420", "car ahead breaking" ),
        ( "000660", "car ahead slowing" ),
        ( "000946", "bend with car ahead" ),
        ( "001035", "taking over car on the right" ),
        ( "001040", "taking over car on the right" ),
        ( "001060", "taking over a car in a bend" ),
        ( "001095", "car ahead breaking" ),
        ( "001160", "car taking over on the left" ),
        ( "001198", "car taking over on the left" ),
    ),
    "SYNTHIA-SEQS-01-FALL": (
        ( "000240", "taking over car on the right" ),
        ( "000325", "car ahead breaking" ),
        ( "000560", "car ahead slowing" ),
        ( "001005", "bend with car ahead" ),
        ( "001135", "taking over car on the right" ),
        ( "001155", "taking over car on the right" ),
        ( "001060", "taking over a car in a bend" ),
        ( "001065", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-01-FOG": (
        ( "000280", "taking over car on the right" ),
        ( "000350", "car ahead breaking" ),
        ( "000495", "car ahead slowing" ),
        ( "000925", "bend with car ahead" ),
        ( "001025", "taking over car on the right" ),
        ( "001035", "taking over car on the right" ),
        ( "001045", "taking over a car in a bend" ),
        ( "001090", "accident ahead" ),
    ),
    "SYNTHIA-SEQS-01-NIGHT": (
        ( "000250", "taking over car on the right" ),
        ( "000335", "car ahead breaking" ),
        ( "000415", "car ahead slowing" ),
        ( "000795", "bend with car ahead" ),
        ( "000870", "taking over car on the right" ),
        ( "000880", "taking over car on the right" ),
        ( "000890", "taking over a car in a bend" ),
        ( "000925", "accident ahead" ),
    ),
    "SYNTHIA-SEQS-01-SPRING": (
        ( "000230", "taking over car on the right" ),
        ( "000280", "car ahead breaking" ),
        ( "000345", "car ahead slowing" ),
        ( "001045", "bend with car ahead" ),
        ( "001140", "taking over car on the right" ),
        ( "001150", "taking over a car in a bend" ),
        ( "001180", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-01-SUMMER": (
        ( "000225", "taking over car on the right" ),
        ( "000275", "car ahead breaking" ),
        ( "000360", "car ahead breaking" ),
        ( "000434", "car taking over on the left" ),
        ( "000835", "bend with car ahead" ),
        ( "000840", "bend with car ahead" ),
        ( "000930", "taking over car on the right" ),
        ( "000935", "taking over car on the right" ),
    ),
    "SYNTHIA-SEQS-01-SUNSET": (
        ( "000235", "taking over car on the right" ),
        ( "000340", "car ahead breaking" ),
        ( "000445", "car taking over on the left" ),
        ( "000865", "bend with car ahead" ),
        ( "000960", "taking over car on the right" ),
        ( "000970", "taking over car on the right" ),
        ( "000975", "taking over a car in a bend" ),
        ( "001015", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-01-WINTER": (
        ( "000245", "taking over car on the right" ),
        ( "000345", "car ahead breaking" ),
        ( "000415", "car taking over on the left" ),
        ( "000420", "car taking over on the left" ),
        ( "000535", "car ahead slowing" ),
        ( "000900", "bend with car ahead" ),
        ( "000905", "bend with car ahead" ),
    ),
    "SYNTHIA-SEQS-01-WINTERNIGHT": (
        ( "000245", "taking over car on the right" ),
        ( "000323", "car taking over on the left" ),
        ( "000335", "car ahead breaking" ),
        ( "000415", "car taking over on the left" ),
        ( "000467", "car ahead slowing" ),
        ( "000830", "bend with car ahead" ),
        ( "000835", "bend with car ahead" ),
        ( "000919", "taking over car on the right" ),
        ( "000928", "taking over a car in a bend" ),
        ( "000945", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-02-DAWN": (
        ( "000000", "car incoming" ),
        ( "000033", "car crossing" ),
        ( "000043", "car crossing" ),
        ( "000090", "bend with lane changes" ),
        ( "000140", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000210", "taking over car on the right" ),
        ( "000230", "taking over a car in a bend" ),
        ( "000390", "car starting ahead" ),
        ( "000465", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-02-FALL": (
        ( "000000", "car incoming" ),
        ( "000033", "car crossing" ),
        ( "000090", "bend with lane changes" ),
        ( "000135", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000215", "taking over car on the right" ),
        ( "000235", "taking over a car in a bend" ),
        ( "000385", "car starting ahead" ),
        ( "000443", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-02-FOG": (
        ( "000000", "car incoming" ),
        ( "000042", "car crossing" ),
        ( "000095", "bend with lane changes" ),
        ( "000145", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000227", "taking over car on the right" ),
        ( "000252", "taking over a car in a bend" ),
        ( "000373", "car starting ahead" ),
        ( "000437", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-02-NIGHT": (
        ( "000001", "car incoming" ),
        ( "000024", "car crossing" ),
        ( "000070", "bend with lane changes" ),
        ( "000107", "car crossing" ),
        ( "000135", "car crossing" ),
        ( "000167", "taking over car on the right" ),
        ( "000185", "taking over a car in a bend" ),
        ( "000295", "car starting ahead" ),
    ),
    "SYNTHIA-SEQS-02-RAINNIGHT": (
        ( "000000", "car incoming" ),
        ( "000024", "car crossing" ),
        ( "000072", "bend with lane changes" ),
        ( "000107", "car crossing" ),
        ( "000135", "car crossing" ),
        ( "000170", "taking over car on the right" ),
        ( "000188", "taking over a car in a bend" ),
        ( "000297", "car starting ahead" ),
    ),
    "SYNTHIA-SEQS-02-SOFTRAIN": (
        ( "000000", "car incoming" ),
        ( "000032", "car crossing" ),
        ( "000097", "bend with lane changes" ),
        ( "000145", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000232", "taking over car on the right" ),
        ( "000258", "taking over a car in a bend" ),
        ( "000388", "car starting ahead" ),
        ( "000460", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-02-SPRING": (
        ( "000000", "car incoming" ),
        ( "000032", "car crossing" ),
        ( "000079", "bend with lane changes" ),
        ( "000120", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000200", "taking over car on the right" ),
        ( "000220", "taking over a car in a bend" ),
        ( "000365", "car starting ahead" ),
        ( "000450", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-02-SUMMER": (
        ( "000000", "car incoming" ),
        ( "000032", "car crossing" ),
        ( "000080", "bend with lane changes" ),
        ( "000125", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000200", "taking over car on the right" ),
        ( "000223", "taking over a car in a bend" ),
        ( "000345", "car starting ahead" ),
    ),
    "SYNTHIA-SEQS-02-SUNSET": (
        ( "000000", "car incoming" ),
        ( "000035", "car crossing" ),
        ( "000102", "bend with lane changes" ),
        ( "000152", "car crossing" ),
        ( "000180", "car crossing" ),
        ( "000248", "taking over car on the right" ),
        ( "000272", "taking over a car in a bend" ),
        ( "000390", "car starting ahead" ),
        ( "000470", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-02-WINTER": (
        ( "000000", "car incoming" ),
        ( "000032", "car crossing" ),
        ( "000087", "bend with lane changes" ),
        ( "000133", "car crossing" ),
        ( "000160", "car crossing" ),
        ( "000215", "taking over car on the right" ),
        ( "000236", "taking over a car in a bend" ),
        ( "000367", "car starting ahead" ),
        ( "000450", "car ahead at roundabout" ),
    ),
    "SYNTHIA-SEQS-04-DAWN": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000070", "parked car on the right" ),
        ( "000110", "bend with lane changes" ),
        ( "000154", "parked car on the left" ),
        ( "000300", "parked car on the right" ),
        ( "000350", "car crossing" ),
        ( "000470", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-FALL": (
        ( "000000", "parked car on the right" ),
        ( "000019", "parked car in front" ),
        ( "000062", "parked car on the right" ),
        ( "000110", "bend with lane changes" ),
        ( "000150", "parked car on the left" ),
        ( "000305", "parked car on the right" ),
        ( "000350", "car crossing" ),
        ( "000490", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-FOG": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000060", "parked car on the right" ),
        ( "000113", "bend with lane changes" ),
        ( "000152", "parked car on the left" ),
        ( "000312", "parked car on the right" ),
        ( "000350", "car crossing" ),
        ( "000515", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-NIGHT": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000060", "parked car on the right" ),
        ( "000115", "bend with lane changes" ),
        ( "000152", "parked car on the left" ),
        ( "000295", "parked car on the right" ),
        ( "000335", "car crossing" ),
        ( "000450", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-RAINNIGHT": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000070", "parked car on the right" ),
        ( "000111", "bend with lane changes" ),
        ( "000151", "parked car on the left" ),
        ( "000297", "parked car on the right" ),
        ( "000335", "car crossing" ),
        ( "000458", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-SOFTRAIN": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000060", "parked car on the right" ),
        ( "000105", "bend with lane changes" ),
        ( "000147", "parked car on the left" ),
        ( "000306", "parked car on the right" ),
        ( "000350", "car crossing" ),
    ),
    "SYNTHIA-SEQS-04-SPRING": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000065", "parked car on the right" ),
        ( "000110", "bend with lane changes" ),
        ( "000152", "parked car on the left" ),
        ( "000307", "parked car on the right" ),
        ( "000350", "car crossing" ),
        ( "000520", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-SUMMER": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000050", "parked car on the right" ),
        ( "000095", "bend with lane changes" ),
        ( "000135", "parked car on the left" ),
        ( "000280", "parked car on the right" ),
        ( "000325", "car crossing" ),
        ( "000470", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-SUNSET": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000070", "parked car on the right" ),
        ( "000105", "bend with lane changes" ),
        ( "000146", "parked car on the left" ),
        ( "000306", "parked car on the right" ),
        ( "000350", "car crossing" ),
        ( "000515", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-04-WINTER": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000050", "parked car on the right" ),
        ( "000100", "bend with lane changes" ),
        ( "000141", "parked car on the left" ),
        ( "000290", "parked car on the right" ),
        ( "000345", "car crossing" ),
    ),
    "SYNTHIA-SEQS-04-WINTERNIGHT": (
        ( "000000", "parked car on the right" ),
        ( "000020", "parked car in front" ),
        ( "000062", "parked car on the right" ),
        ( "000113", "bend with lane changes" ),
        ( "000152", "parked car on the left" ),
        ( "000292", "parked car on the right" ),
        ( "000330", "car crossing" ),
        ( "000460", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-DAWN": (
        ( "000070", "car ahead slowing" ),
        ( "000075", "taking over car on the left" ),
        ( "000090", "car crossing" ),
        ( "000280", "car crossing" ),
        ( "000395", "roundabout" ),
        ( "000560", "incoming car" ),
        ( "000575", "incoming car" ),
        ( "000745", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-FALL": (
        ( "000070", "car ahead slowing" ),
        ( "000075", "taking over car on the left" ),
        ( "000090", "car crossing" ),
        ( "000270", "car crossing" ),
        ( "000391", "roundabout" ),
        ( "000564", "incoming car" ),
        ( "000574", "incoming car" ),
        ( "000745", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-FOG": (
        ( "000065", "car ahead slowing" ),
        ( "000068", "taking over car on the left" ),
        ( "000079", "car crossing" ),
        ( "000250", "car crossing" ),
        ( "000360", "roundabout" ),
        ( "000540", "incoming car" ),
        ( "000560", "incoming car" ),
        ( "000680", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-NIGHT": (
        ( "000056", "car ahead slowing" ),
        ( "000061", "taking over car on the left" ),
        ( "000072", "car crossing" ),
        ( "000280", "car crossing" ),
        ( "000350", "roundabout" ),
        ( "000500", "incoming car" ),
        ( "000511", "incoming car" ),
        ( "000663", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-RAIN": (
        ( "000070", "car ahead slowing" ),
        ( "000076", "taking over car on the left" ),
        ( "000090", "car crossing" ),
        ( "000265", "car crossing" ),
        ( "000358", "roundabout" ),
        ( "000530", "incoming car" ),
        ( "000537", "incoming car" ),
        ( "000680", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-RAINNIGHT": (
        ( "000057", "car ahead slowing" ),
        ( "000063", "taking over car on the left" ),
        ( "000070", "car crossing" ),
        ( "000247", "car crossing" ),
        ( "000350", "roundabout" ),
        ( "000502", "incoming car" ),
        ( "000509", "incoming car" ),
        ( "000668", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-SOFTRAIN": (
        ( "000070", "car ahead slowing" ),
        ( "000076", "taking over car on the left" ),
        ( "000085", "car crossing" ),
        ( "000270", "car crossing" ),
        ( "000392", "roundabout" ),
        ( "000553", "incoming car" ),
        ( "000579", "incoming car" ),
        ( "000730", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-SPRING": (
        ( "000068", "car ahead slowing" ),
        ( "000073", "taking over car on the left" ),
        ( "000081", "car crossing" ),
        ( "000260", "car crossing" ),
    ),
    "SYNTHIA-SEQS-05-SUMMER": (
        ( "000066", "car ahead slowing" ),
        ( "000070", "taking over car on the left" ),
        ( "000085", "car crossing" ),
        ( "000261", "car crossing" ),
        ( "000352", "roundabout" ),
        ( "000506", "incoming car" ),
        ( "000524", "incoming car" ),
        ( "000536", "incoming car" ),
        ( "000672", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-SUNSET": (
        ( "000079", "taking over car on the left" ),
        ( "000087", "car crossing" ),
        ( "000256", "car crossing" ),
        ( "000369", "roundabout" ),
        ( "000532", "incoming car" ),
        ( "000537", "incoming car" ),
        ( "000693", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-WINTER": (
        ( "000070", "car ahead slowing" ),
        ( "000076", "taking over car on the left" ),
        ( "000085", "car crossing" ),
        ( "000260", "car crossing" ),
        ( "000371", "roundabout" ),
        ( "000530", "incoming car" ),
        ( "000538", "incoming car" ),
        ( "000545", "incoming car" ),
        ( "000695", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-05-WINTERNIGHT": (
        ( "000054", "car ahead slowing" ),
        ( "000060", "taking over car on the left" ),
        ( "000077", "car crossing" ),
        ( "000258", "car crossing" ),
        ( "000350", "roundabout" ),
        ( "000505", "incoming car" ),
        ( "000520", "incoming car" ),
        ( "000665", "car ahead breaking" ),
    ),
    "SYNTHIA-SEQS-06-DAWN": (
        ( "000020", "car ahead slowing" ),
        ( "000035", "car crossing" ),
        ( "000085", "car taking over on the right" ),
        ( "000115", "car ahead breaking" ),
        ( "000314", "taking over car on the left" ),
        ( "000355", "taking over car on the right" ),
        ( "000371", "taking over car on the right" ),
        ( "000635", "bend with car ahead" ),
        ( "000721", "car taking over on the left" ),
    ),
    "SYNTHIA-SEQS-06-FOG": (
        ( "000018", "car ahead slowing" ),
        ( "000035", "car crossing" ),
        ( "000084", "car taking over on the right" ),
        ( "000099", "car ahead breaking" ),
        ( "000242", "taking over car on the left" ),
        ( "000281", "taking over car on the right" ),
        ( "000289", "taking over car on the right" ),
        ( "000610", "bend with car ahead" ),
        ( "000642", "car taking over on the left" ),
    ),
    "SYNTHIA-SEQS-06-NIGHT": (
        ( "000018", "car ahead slowing" ),
        ( "000065", "car ahead breaking" ),
        ( "000220", "taking over car on the left" ),
        ( "000260", "taking over car on the right" ),
        ( "000275", "taking over car on the right" ),
        ( "000530", "bend with car ahead" ),
    ),
    "SYNTHIA-SEQS-06-SPRING": (
        ( "000020", "car ahead slowing" ),
        ( "000040", "car crossing" ),
        ( "000096", "car taking over on the right" ),
        ( "000116", "car ahead breaking" ),
        ( "000344", "taking over car on the left" ),
        ( "000355", "taking over car on the right" ),
        ( "000383", "taking over car on the right" ),
        ( "000784", "bend with car ahead" ),
    ),
    "SYNTHIA-SEQS-06-SUMMER": (
        ( "000020", "car ahead slowing" ),
        ( "000040", "car crossing" ),
        ( "000090", "car taking over on the right" ),
        ( "000114", "car ahead breaking" ),
        ( "000338", "taking over car on the left" ),
        ( "000373", "taking over car on the right" ),
        ( "000395", "taking over car on the right" ),
        ( "000770", "bend with car ahead" ),
    ),
    "SYNTHIA-SEQS-06-SUNSET": (
        ( "000020", "car ahead slowing" ),
        ( "000074", "car taking over on the right" ),
        ( "000097", "car ahead breaking" ),
        ( "000248", "taking over car on the left" ),
        ( "000285", "taking over car on the right" ),
        ( "000307", "taking over car on the right" ),
        ( "000633", "bend with car ahead" ),
    ),
    "SYNTHIA-SEQS-06-WINTER": (
        ( "000020", "car ahead slowing" ),
        ( "000035", "car crossing" ),
        ( "000082", "car taking over on the right" ),
        ( "000110", "car ahead breaking" ),
        ( "000271", "taking over car on the left" ),
        ( "000325", "taking over car on the right" ),
        ( "000354", "taking over car on the right" ),
        ( "000632", "bend with car ahead" ),
    ),
    "SYNTHIA-SEQS-06-WINTERNIGHT": (
        ( "000020", "car ahead slowing" ),
        ( "000033", "car crossing" ),
        ( "000086", "car taking over on the right" ),
        ( "000097", "car ahead breaking" ),
        ( "000249", "taking over car on the left" ),
        ( "000287", "taking over car on the right" ),
        ( "000314", "taking over car on the right" ),
        ( "000656", "car taking over on the left" ),
    )
}
