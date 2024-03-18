package highCL

import (
	pure "github.com/opencl-pure/pureCL"
	"strings"
)

type BuildOptions struct {
	// Preprocessor options
	Warnings          bool
	Macros            map[string]string
	DirectoryIncludes []string
	Version           pure.Version
	// Math intrinsics options
	SinglePrecisionConstant bool
	MadEnable               bool
	NoSignedZeros           bool
	FastRelaxedMaths        bool
	UnsafeMaths             bool
	// Extensions
	NvidiaVerbose bool
}

func (op *BuildOptions) String() string {
	if op == nil {
		return ""
	}
	var sb strings.Builder
	// Preprocessor
	if op.Warnings {
		sb.WriteString("-w")
		sb.WriteRune(' ')
	}
	if op.Version != "" {
		sb.WriteString("-cl-std=" + string(op.Version))
		sb.WriteRune(' ')
	}
	// Math intrinsics
	if op.SinglePrecisionConstant {
		sb.WriteString("-cl-single-precision-constant")
		sb.WriteRune(' ')
	}
	if op.MadEnable {
		sb.WriteString("-cl-mad-enable")
		sb.WriteRune(' ')
	}
	if op.NoSignedZeros {
		sb.WriteString("-cl-no-signed-zeros")
		sb.WriteRune(' ')
	}
	if op.FastRelaxedMaths {
		sb.WriteString("-cl-fast-relaxed-math")
		sb.WriteRune(' ')
	}
	if op.UnsafeMaths {
		sb.WriteString("-cl-unsafe-math-optimizations")
		sb.WriteRune(' ')
	}
	// Extensions
	if op.NvidiaVerbose {
		sb.WriteString("-cl-nv-verbose")
		sb.WriteRune(' ')
	}

	return sb.String()
}
