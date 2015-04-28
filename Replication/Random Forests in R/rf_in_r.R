###### Random Forests in R Blog Post ######
######################################

rm(list=ls())

pkgs = list('randomForest','party','randomForestSRC','foreign')
invisible(lapply(pkgs, library, character.only=TRUE, quietly=TRUE))

setwd('~/bensfisher.github.io/Replication/Random Forests in R/')

bkz = read.dta('BKZ.dta')
bkz$disp = as.factor(bkz$disp)
bkz$logpy = log(bkz$py +1)
bkz$contig = as.factor(bkz$contig)
fmla = formula(as.factor(disp) ~ aysm + contig + ally + sq + dema + demb + logpy) 

### fit models and time ###
ptm = proc.time()
rf.mod = randomForest(fmla, bkz, ntree=500, mtry=3, importance=TRUE)
rf.time = proc.time() - ptm

ptm = proc.time()
cf.mod = cforest(fmla, bkz, controls=cforest_unbiased(ntree=500, mtry=3))
cf.time = proc.time() - ptm

ptm = proc.time()
src.mod = rfsrc(fmla, bkz, ntree=500, mtry=3, importance=c('permute.ensemble'), var.used='all.trees')
src.time = proc.time() - ptm

packages = c('randomForest', 'party', 'randomForestSRC')
time = c(rf.time, cf.time, src.time)
barplot(time, names=packages, xlab="Package", ylab="Time in seconds", ylim=c(0,900), main="Speed Comparison")

### variables used ###
rf.vars = varUsed(rf.mod)
cf.mod = src.mod$var.used

count = function(forest, inames = NULL) {
	if (is.null(inames) && extends(class(forest), "RandomForest"))
	inames = names(forest@data@get("input"))
	resultvec = rep(0, length(inames))
	
	myfunc = function(x, inames, resultvec) {
		names(x) = c('nodeID', 'weights', 'criterion', 'terminal',
				'psplit', 'ssplits', 'prediciton', 'left', 'right')
		names(x$criterion) = c('statistic', 'criterion', 'maxcriterion')
		if (!x$terminal) {
			resultvec[x$psplit[[1]]] = resultvec[x$psplit[[1]]] + 1
			resultvec = myfunc(x$left, inames = inames, resultvec = resultvec)
			resultvec = myfunc(x$right, inames = inames, resultvec = resultvec)
			}
		return(resultvec)
		}
	for (k in 1:length(forest@ensemble)) {
		resultvec = myfunc(forest@ensemble[[k]], inames, resultvec)
		}
names(resultvec) = inames
return(resultvec) 
}

environment(count) <- environment(varimp)

### Variable Importance ###

ptm = proc.time()
rf.imp = importance(rf.mod)
proc.time() - ptm
varImpPlot(rf.mod, type=1, scale=FALSE)

ptm = proc.time()
cf.imp = varimp(cf.mod, OOB=TRUE)
proc.time() - ptm
barplot(cf.imp)


### Partial Plots ###
partialPlot(rf.mod, bkz, x.var=contig, which.class=1)

ptm = proc.time()
plot.variable(src.mod, 'contig', which.outcome='1', partial=TRUE)
proc.time()-ptm

ptm = proc.time()
plot.variable(src.mod, 'logpy', which.outcome='1', partial=TRUE)
proc.time() - ptm


