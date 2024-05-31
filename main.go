package main

import (
	"fmt"
	"math"
)

var RVector [6]float64 = [6]float64{60, 35, 37, 34, 41, 50}
var PiVector [6]float64 = [6]float64{6, 7, 8, 8, 7, 6}
var UnitRevenue [6]float64 = [6]float64{12, 11, 10, 8, 9, 10}
var convergeFlags [6]bool = [6]bool{false, false, false, false, false, false}

const S float64 = 600
const D float64 = 0.01
const K float64 = 5
const T float64 = 60
const TUL float64 = 0.0038
const TDL float64 = 0.0020
const e0 float64 = 9.82
const e1 float64 = 4.26
const uploadCost float64 = 0.0000000232
const downloadCost float64 = 0.0000000232
const investmentCost float64 = 0.003666
const minR float64 = 0
const maxR float64 = 100      //Max of R vector // not effecting the output
const penality float64 = 0.1  //effects the output
const stepsize float64 = 0.1  //effects the output
const threshold float64 = 0.1 //effects the output

func main() {
	j := 0
	for {
		globalConv := true
		fmt.Println("round = ", j)
		for i, _ := range RVector {
			r := getR(i)                          //Step 6
			newR := getNewR(r, RVector[i])        //Step 7
			newPi := getNewPi(PiVector[i], i)     //Step 8
			if (math.Abs(newR - r)) > threshold { //Step 9
				RVector[i] = newR
				PiVector[i] = newPi
			} else {
				convergeFlags[i] = true
			}
		}
		fmt.Println(convergeFlags)
		fmt.Println(globalConv)

		j++
		for _, flag := range convergeFlags {
			if !flag {
				globalConv = false
				break
			}
		}
		if globalConv {
			fmt.Println("All flags are set to true. Stopping the loop.")
			break
		}
		fmt.Println("RVector = ", RVector)
		fmt.Println("PiVector= ", PiVector)
	}
}
func getUtility(rF float64, organisation int) float64 {
	er0 := e0 / e1
	erF := e0 / (e1 + (K * rF))
	return UnitRevenue[organisation] * (er0 - erF)
}
func getCost(rF float64, organisation int) float64 {
	return getCommunicationCost(rF) + getInvestmentCost(rF, organisation) + getOperatingCost(rF, organisation)
}

func getCommunicationCost(rF float64) float64 {
	return (uploadCost + downloadCost) * rF
}

func getInvestmentCost(rF float64, organisation int) float64 {
	return investmentCost * getFNought(rF, organisation)
}

func getOperatingCost(rF float64, organisation int) float64 {
	return 0.00004833 * getMaxTime(rF, organisation) * getFNought(rF, organisation) * getFNought(rF, organisation) * S * D * K * rF
}

func getR(organisation int) float64 {
	maxPayOff := 0.0
	maxPayAt := 0.0
	for rF := minR; rF <= maxR; rF++ {
		payOff := getUtility(rF, organisation) - getCost(rF, organisation) //- penality*getPenalityTerm()
		if payOff > maxPayOff {
			maxPayOff = payOff
			maxPayAt = rF
		}
	}
	fmt.Printf("Max PayOff is %f, and max Pay at %f \n", maxPayOff, maxPayAt)
	return maxPayAt
}
func getNewR(rComputed float64, rOld float64) float64 {
	newR := rOld + penality*(rComputed-rOld)
	return newR
}
func getNewPi(oldPi float64, n int) float64 {
	i := n - 2
	j := n - 1
	if n == 0 {
		i = len(RVector) - 2
		j = len(RVector) - 1
	} else if n == 1 {
		i = len(RVector) - 1
	}
	newPi := oldPi + penality*stepsize*(RVector[i]-RVector[j])
	return newPi
}
func getRAverage(rF float64, organisation int) float64 {
	sum := 0.0
	RvectorTemp := RVector
	RvectorTemp[organisation] = rF
	for _, num := range RvectorTemp {
		sum = sum + num
	}
	return float64(sum) / float64(len(RVector))
}
func getFNought(rF float64, organisation int) float64 {
	return S * D * K / ((T / getRAverage(rF, organisation)) - TUL - TDL)
}

func getMaxTime(rF float64, organisation int) float64 {
	return ((S*D*K)/getFNought(rF, organisation) + TUL + TDL)
}

func getPenalityTerm() float64 {
	sum := 0.0
	for i, _ := range RVector {
		j := 0
		k := 0
		if i == 0 {
			j = len(RVector) - 2
			k = len(RVector) - 1
		} else if i == 1 {
			j = len(RVector) - 1
		} else {
			j = i - 2
			k = i - 1
		}
		temp := RVector[j] - RVector[k]
		sum += temp * temp
	}
	return sum
}

// For every organisation
// 1. payoff graph
// 2. rF graph
// 3. pi graph
