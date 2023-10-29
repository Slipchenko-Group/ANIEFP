  QCDIR=/depot/lslipche/apps/qchem; export QCDIR
  echo $QC
  QC=${QCDIR}/QCHEM_5.2.1/; export QC
  QCMPI=openmpi; export QCMPI
  PATH=${PATH}:${QCDIR}/QCHEM_5.2.1/bin:${QCDIR}/QCHEM_5.2.1/bin/perl; export PATH
  QCRSH=ssh; export QCRSH
  QCAUX=/depot/lslipche/apps/qchem/qcaux; export QCAUX
  export QCREF=/group/lslipche/apps/qcref
  QCSCRATCH=${RCAC_SCRATCH}; export QCSCRATCH
  QCDIR=/depot/lslipche/apps/qchem; export QCDIR
  echo $QC
