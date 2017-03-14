package bdv.img.huygens;

import java.io.File;
import java.util.HashMap;
import java.util.List;

import bdv.AbstractViewerSetupImgLoader;
import bdv.ViewerImgLoader;
import bdv.ViewerSetupImgLoader;
import bdv.cache.CacheControl;
import bdv.cache.CacheHints;
import bdv.cache.LoadingStrategy;
import bdv.img.cache.CacheArrayLoader;
import bdv.img.cache.CachedCellImg;
import bdv.img.cache.VolatileGlobalCellCache;
import bdv.img.cache.VolatileImgCells;
import bdv.img.cache.VolatileImgCells.CellCache;
import ch.systemsx.cisd.base.mdarray.MDFloatArray;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import mpicbg.spim.data.generic.sequence.AbstractSequenceDescription;
import mpicbg.spim.data.generic.sequence.BasicViewSetup;
import mpicbg.spim.data.generic.sequence.ImgLoaderHint;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.basictypeaccess.volatiles.array.VolatileFloatArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.volatiles.VolatileFloatType;
import net.imglib2.util.Fraction;

public class HuygensImageLoader implements ViewerImgLoader
{
	final File hdf5File;

	final IHDF5Reader hdf5Reader;

	final AbstractSequenceDescription< ?, ?, ? > sequenceDescription;

	final String datasetPath;

	final VolatileGlobalCellCache cache;

	final HashMap< Integer, HuygensSetupImageLoader > setupImgLoaders;

	public HuygensImageLoader(
			final File hdf5File,
			final AbstractSequenceDescription< ?, ?, ? > sequenceDescription,
			final String datasetPath,
			final long[] dimensions,
			final int[] cellDimensions )
	{
		this.hdf5File = hdf5File;
		this.datasetPath = datasetPath;
		this.sequenceDescription = sequenceDescription;
		this.hdf5Reader = HDF5Factory.openForReading( hdf5File );

		final double[] resolutions = new double[] { 1, 1, 1 };

		final HuygensCacheArrayLoader loader = new HuygensCacheArrayLoader();
		cache = new VolatileGlobalCellCache( 1, 1 );

		setupImgLoaders = new HashMap<>();
		final List< ? extends BasicViewSetup > setups = sequenceDescription.getViewSetupsOrdered();
		for ( final BasicViewSetup setup : setups )
		{
			final int setupId = setup.getId();
			setupImgLoaders.put( setupId, new HuygensSetupImageLoader(
					setupId, dimensions, cellDimensions, resolutions, loader, cache ) );
		}
	}

	@Override
	public ViewerSetupImgLoader< ?, ? > getSetupImgLoader( final int setupId )
	{
		return setupImgLoaders.get( setupId );
	}

	@Override
	public CacheControl getCacheControl()
	{
		return cache;
	}

	class HuygensCacheArrayLoader implements CacheArrayLoader< VolatileFloatArray >
	{
		private VolatileFloatArray theEmptyArray = new VolatileFloatArray( 1000, false );

		@Override
		public VolatileFloatArray loadArray(
				final int timepoint,
				final int setup,
				final int level,
				final int[] dimensions,
				final long[] min ) throws InterruptedException
		{
			final int[] dim = new int[ 5 ];
			dim[ 0 ] = 1; // C
			dim[ 1 ] = 1; // T
			dim[ 2 ] = dimensions[ 2 ]; // Z
			dim[ 3 ] = dimensions[ 1 ]; // Y
			dim[ 4 ] = dimensions[ 0 ]; // X

			final long[] offset = new long[ 5 ];
			offset[ 0 ] = setup; // C
			offset[ 1 ] = timepoint; // T
			offset[ 2 ] = min[ 2 ]; // Z
			offset[ 3 ] = min[ 1 ]; // Y
			offset[ 4 ] = min[ 0 ]; // X

			final MDFloatArray mdarray = hdf5Reader.float32().readMDArrayBlockWithOffset( datasetPath, dim, offset );
			return new VolatileFloatArray( mdarray.getAsFlatArray(), true );
		}

		@Override
		public int getBytesPerElement()
		{
			return 4;
		}

		@Override
		public VolatileFloatArray emptyArray( final int[] dimensions )
		{
			int numEntities = 1;
			for ( int i = 0; i < dimensions.length; ++i )
				numEntities *= dimensions[ i ];
			if ( theEmptyArray.getCurrentStorageArray().length < numEntities )
				theEmptyArray = new VolatileFloatArray( numEntities, false );
			return theEmptyArray;
		}
	}

	class HuygensSetupImageLoader extends AbstractViewerSetupImgLoader< FloatType, VolatileFloatType >
	{
		final double[][] resolutions;

		final AffineTransform3D[] mipmapTransforms;

		final CacheArrayLoader< VolatileFloatArray > loader;

		final int setupId;

		final long[] dimensions;

		final int[] cellDimensions;

		public HuygensSetupImageLoader(
				final int setupId,
				final long[] dimensions,
				final int[] cellDimensions,
				final double[] resolutions,
				final CacheArrayLoader< VolatileFloatArray > loader,
				final VolatileGlobalCellCache cache )
		{
			super( new FloatType(), new VolatileFloatType() );
			this.setupId = setupId;
			this.dimensions = dimensions;
			this.cellDimensions = cellDimensions;
			this.resolutions = new double[][] { resolutions };
			this.loader = loader;
			final AffineTransform3D mipmapTransform = new AffineTransform3D();
			mipmapTransform.set( resolutions[ 0 ], 0, 0 );
			mipmapTransform.set( resolutions[ 1 ], 1, 1 );
			mipmapTransform.set( resolutions[ 2 ], 2, 2 );
			this.mipmapTransforms = new AffineTransform3D[] { mipmapTransform };
		}

		@Override
		public RandomAccessibleInterval< VolatileFloatType > getVolatileImage( final int timepointId, final int level, final ImgLoaderHint... hints )
		{
			final CachedCellImg< VolatileFloatType, VolatileFloatArray >  img = prepareCachedImage( timepointId, level, LoadingStrategy.BLOCKING );
			final VolatileFloatType linkedType = new VolatileFloatType( img );
			img.setLinkedType( linkedType );
			return img;
		}

		@Override
		public RandomAccessibleInterval< FloatType > getImage( final int timepointId, final int level, final ImgLoaderHint... hints )
		{
			final CachedCellImg< FloatType, VolatileFloatArray >  img = prepareCachedImage( timepointId, level, LoadingStrategy.BLOCKING );
			final FloatType linkedType = new FloatType( img );
			img.setLinkedType( linkedType );
			return img;
		}

		@Override
		public double[][] getMipmapResolutions()
		{
			return resolutions;
		}

		@Override
		public AffineTransform3D[] getMipmapTransforms()
		{
			return mipmapTransforms;
		}

		@Override
		public int numMipmapLevels()
		{
			return 1;
		}

		protected < T extends NativeType< T > > CachedCellImg< T, VolatileFloatArray > prepareCachedImage(
				final int timepointId, final int level,
				final LoadingStrategy loadingStrategy )
		{
			final int priority = 0;
			final CacheHints cacheHints = new CacheHints( loadingStrategy, priority, false );
			final CellCache< VolatileFloatArray > c = cache.new VolatileCellCache<>( timepointId, setupId, level, cacheHints, loader );
			final VolatileImgCells< VolatileFloatArray > cells = new VolatileImgCells<>( c, new Fraction(), dimensions, cellDimensions );
			final CachedCellImg< T, VolatileFloatArray > img = new CachedCellImg<>( cells );
			return img;
		}
	}
}
