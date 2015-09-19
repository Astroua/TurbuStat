
# Read in cmd line args
# Should contain 1) the path, 2) # of iterations
args = commandArgs(TRUE)

startTime = Sys.time()
# setwd(args[1])

FidDes00 = read.csv('distances_0_0.csv', header = T)
FidFid00 = read.csv('fiducials_0_0.csv', header = T)

FidDes22 = read.csv('distances_2_2.csv', header = T)
FidFid22 = read.csv('fiducials_2_2.csv', header = T)

# Remove unneeded columns
FidDes00$X = FidDes00$Fiducial = FidDes00$Designs = FidDes00$Plasma.Beta = FidDes00$k = FidDes00$Mach.Number = FidDes00$Solenoidal.Fraction = FidDes00$Virial.Parameter = NULL
FidDes22$X = FidDes22$Fiducial = FidDes22$Designs = FidDes22$Plasma.Beta = FidDes22$k = FidDes22$Mach.Number = FidDes22$Solenoidal.Fraction = FidDes22$Virial.Parameter = NULL

FidFid00$X = FidFid00$Fiducial.1 = FidFid00$Fiducial.2 = NULL
FidFid22$X = FidFid22$Fiducial.1 = FidFid22$Fiducial.2 = NULL

y = rbind(FidFid00, FidFid22, FidDes00, FidDes22)
x = c(rep(0, length(FidFid00[,1])),
	  rep(0, length(FidFid22[,1])),
	  rep(1, length(FidDes00[,1])),
	  rep(1, length(FidDes22[,1])))

nperm = as.numeric(args[2])
nstats = length(FidDes00)
pVals = rep(0, nstats)

for(i in 1:nperm)
{
	ys = y[sample(nrow(y)),]
	for(j in 1:nstats)
	{
		permdata = cbind(ys[j], x)
		data = cbind(y[j], x)
		names(data) = names(permdata) = c("y", "x")
		if(summary(lm(y ~ x, data = permdata))$coefficients[2] > summary(lm(y ~ x, data = data))$coefficients[2])
		{
			pVals[j] = pVals[j] + 1
		}
	}
	if (i %% nperm/10 == 0)
	{
		print(paste(i, '/', nperm, sep=""))
	}
}
names(pVals) = names(FidDes00)
pVals = pVals/nperm
print(pVals)
print(Sys.time() - startTime)
write.table(pVals, file=paste("pValues", nperm, "face_00_22.csv"), sep=",", col.names = F, row.names=T)
