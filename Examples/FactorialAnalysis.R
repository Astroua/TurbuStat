library(lme4)
library(xtable)
library(MASS)
###########

resFace0=read.table('distances_0_0.csv',header=T,sep=',')
resFace2=read.table('distances_2_2.csv',header=T,sep=',')

# resFace0=resFace0[-c(1:32),]
# resFace2=resFace2[-c(1:32),]

##Create Indicator for which Fiducial
Cube=rep(rep(c(1:5),each=32),2)
##Create indicator for the Face
fc=rep(c(-1,1),each=160)
###############
factors=c('Solenoidal.Fraction','Virial.Parameter','k', 'Mach.Number','Plasma.Beta')
des0=resFace0[,factors]

colnames(des0)=c('sf','vp','k','m','pb')
res0=resFace0[,!(names(resFace0)%in%c('Ind',factors,'Fiducial','Designs', 'X'))]
stats=colnames(res0)
#des0=as.numeric(des0)
#res0=as.numeric(res0)


des2=resFace2[,factors]
colnames(des2)=c('sf','vp','k','m','pb')
res2=resFace2[,!(names(resFace2)%in%c('Ind',factors,'Fiducial','Designs', 'X'))]
#des2=as.numeric(des2)
#res2=as.numeric(res2)


data=cbind(Cube,fc,rbind(des0,des2),rbind(res0,res2))

apply(data,2,mean)

##############
numStat=length(stats)
indStat=1:numStat

TVals=matrix(0,nrow=63,ncol=numStat)

for(i in 1:numStat)
{
    print(stats[i])
    mod=paste(stats[i],'~fc*sf*vp*k*m*pb+(1|Cube)',sep='')
    ReMod=lmer(mod,REML=F,data=data)

    TVals[,i]=matrix(summary(ReMod)$coeff[-1,3,drop=F])
}

TVals=data.frame(TVals)
colnames(TVals)=stats
rownames(TVals)=rownames(summary(ReMod)$coeff[-1,3,drop=F])

write.table(TVals,'ResultsFactorial.csv',sep=',')
write.table(data, 'DataforFits.csv', sep=',')
