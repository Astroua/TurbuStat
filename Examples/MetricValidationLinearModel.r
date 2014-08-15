startTime = Sys.time()
setwd('~/Dropbox/AstroStatistics/Full\ Factorial//Full\ Results')
importFidDes = read.csv('distances_0_0.csv', header = T)
importFidFid = read.csv('fiducials_0_0.csv', header = T)
FidDes = importFidDes
FidDes$X = FidDes$Fiducial = FidDes$Designs = FidDes$Plasma.Beta = FidDes$k = FidDes$Mach.Number = FidDes$Solenoidal.Fraction = FidDes$Virial.Parameter = NULL
FidFid = importFidFid
FidFid$X = FidFid$Fiducial.1 = FidFid$Fiducial.2 = NULL
y = rbind(FidFid,FidDes)
x = c(0 * 1:(length(FidFid[,1])), 1 + 0 * 1:length(FidDes[,1]))
repeatVal = 500000
pVals = c(0*1:length(FidDes))
for(i in 1:repeatVal)
{
	ys = y[sample(nrow(y)),]
	for(j in 1:length(names(FidDes)))
	{
		permdata = cbind(ys[j], x)
		data = cbind(y[j], x)
		names(data) = names(permdata) = c("y", "x")
		if(summary(lm(y ~ x, data = permdata))$coefficients[2] > summary(lm(y ~ x, data = data))$coefficients[2])
		{
			pVals[j] = pVals[j] + 1
		}
	}
	if(i %% 1000 == 0)
	{
		print(paste(i, '/', repeatVal, sep=""))
	}
}
names(pVals) = names(FidDes)
pVals = pVals/repeatVal
print(pVals)
print(Sys.time() - startTime)
write.table(pVals, file=paste("pValues", repeatVal, "face_00.csv"), sep=",", col.names = F, row.names=T)
